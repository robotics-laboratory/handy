import tqdm
import wandb
import torch
import numpy as np
import torchvision.transforms as T

from random import shuffle

from utils.wandb import WanDBWriter, MetricTracker
from metrics import IoU


class Trainer:
    def __init__(self, model, criterion, optimizer, device, config, dataloaders, metrics, scheduler = None):
        self.device = device
        self.config = config
        self.logger = config.get_logger("trainer", config["trainer"]["verbosity"])

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler

        self.epochs = config["epochs"]
        self.save_period = config["save_period"]
        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir
        self.writer = WanDBWriter(config, self.logger)

        self.train_data_loader = dataloaders["train"]
        self.eval_dataloader = {k: v for k, v in dataloaders.items() if k != "train"}
        self.len_epoch = len(self.train_data_loader)

        self.train_metrics = MetricTracker("loss", *[m.name for m in metrics], writer=self.writer)
        self.eval_metrics = MetricTracker("loss", *[m.name for m in metrics], writer=self.writer)

    def train(self):
        try:
            self._train_process()
        except KeyboardInterrupt as e:
            self.logger.info("Saving model on keyboard interrupt")
            self._save_checkpoint(self._last_epoch, save_best=False)
            raise e

    def _train_process(self):
        """
        Full training logic
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            self._last_epoch = epoch
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {"epoch": epoch}
            log.update(result)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info("    {:15s}: {}".format(str(key), value))

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch)
    
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)
        for batch_idx, batch in enumerate(
                tqdm(self.train_dataloader, desc="train", total=self.len_epoch)
        ):
            try:
                batch = self.process_batch(
                    batch,
                    is_train=True,
                    metrics=self.train_metrics,
                )
            except RuntimeError as e:
                if "out of memory" in str(e):
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e

            if batch_idx == self.len_epoch - 1:
                self.writer.set_step(epoch * self.len_epoch + batch_idx)
                self.writer.add_scalar(
                    "learning rate", self.lr_scheduler.get_last_lr()[0]
                )
                self._log_predictions(**batch, is_train=True)
                self._log_scalars(self.train_metrics)

                last_train_metrics = self.train_metrics.result()
                
        log = last_train_metrics

        for part, dataloader in self.evaluation_dataloaders.items():
            val_log = self._evaluation_epoch(epoch, part, dataloader)
            log.update(**{f"{part}_{name}": value for name, value in val_log.items()})

        return log
    
    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.eval_metrics.reset()
        predictions = []
        labels = []

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                    enumerate(dataloader),
                    desc=part,
                    total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch,
                    is_train=False,
                    metrics=self.eval_metrics,
                )
                predictions.extend(batch["preds"].cpu()[:, 0].tolist())
                labels.extend(batch["labels"].cpu().tolist())
            
            for metric in self.metrics:
                self.evaluation_metrics.update(metric.name, metric(np.array(predictions), np.array(labels)))

            self.writer.set_step(epoch * self.len_epoch, part)
            self._log_scalars(self.evaluation_metrics)
            self._log_predictions(**batch, is_train=False)

        return self.eval_metrics.result()

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the HPU
        """
        for tensor_for_gpu in ["image", "bbox", "mark"]:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch
    
    def process_batch(self, batch, is_train, metrics):
        """
        Process a batch of data
        """
        batch = self.move_batch_to_device(batch, self.device)
        if is_train:
            self.optimizer.zero_grad()
        outputs = self.model(**batch)
        batch.update(outputs)
        loss = self.criterion(**batch)
        if is_train:
            loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
        batch.update({"loss": loss.item()})
        metrics.update("loss", loss.item())

        return batch
    
    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))
    
    def _log_predictions(self, image, bbox, mark, pred_bbox, pred_mark, examples_tp_log=5):
        if self.writer is None:
            return
        results = list(zip(image, bbox, mark, pred_bbox, pred_mark))
        shuffle(results)
        rows = {}
        for i, image, bbox, mark, pred_bbox, pred_mark in enumerate(results[:examples_tp_log]):
            pil_image = T.ToPILImage()(image.cpu())
            wandb_image = wandb.Image(pil_image, 
                                      caption=f"GT: {mark}, Pred: {pred_mark}",
                                      boxes={
                                            "predictions": {
                                                "box_data": [{
                                                    "position": {
                                                        "minX": pred_bbox[0],
                                                        "maxX": pred_bbox[2],
                                                        "minY": pred_bbox[1],
                                                        "maxY": pred_bbox[3]
                                                    },
                                                    "class_id": 0,
                                                    "box_caption": "prdicted ball",
                                                }
                                                ]
                                            },
                                            "ground_truth": {
                                                "box_data": [{
                                                    "position": {
                                                        "minX": bbox[0],
                                                        "maxX": bbox[2],
                                                        "minY": bbox[1],
                                                        "maxY": bbox[3]
                                                    },
                                                    "class_id": 1,
                                                    "box_caption": "ground truth ball",
                                                }
                                                ]
                                            }
                                        })
            rows[f"image_{i}"] = {
                "image": wandb_image,
                "gt": mark,
                "pred": pred_mark,
                "iou": IoU()(bbox, pred_bbox)
            }
        self.writer.add_table("predictions", rows)
    
    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        """
        state = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "config": self.config,
        }
        filename = str(self.checkpoint_dir / "checkpoint.pth")
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / "model_best.pth")
            torch.save(state, best_path)
            self.logger.info("Saving current best: {} ...".format(best_path))