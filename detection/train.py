import lightning as L
import torch
import argparse
from datetime import datetime
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor


from model import get_model
from datasets import DetectionDataModule

class LitDetector(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.val_map = MeanAveragePrecision()
        self.val_map.warn_on_many_detections = False
    
    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self.model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        self.log('train_loss', loss, on_step=True, on_epoch=True, batch_size=len(images))
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.model(images, targets)

        preds = []
        target = []
        for i in range(len(images)):
            true_dict = dict()
            preds_dict = dict()
            true_dict['boxes'] = targets[i]['boxes'].detach().cpu()
            true_dict['labels'] = targets[i]['labels'].detach().cpu()
            preds_dict['boxes'] = outputs[i]['boxes'].detach().cpu()
            preds_dict['scores'] = outputs[i]['scores'].detach().cpu()
            preds_dict['labels'] = outputs[i]['labels'].detach().cpu()
            preds.append(preds_dict)
            target.append(true_dict)
                
        self.val_map.update(preds, target)
    
    def on_validation_epoch_end(self):
        mAP_result = self.val_map.compute()
        self.log('mAP_small', mAP_result['map_small'])
        self.log('mAP@0.50', mAP_result['map_50'])
        self.log('mAP@0.50:0.95', mAP_result['map'])
        self.log('mAR_small', mAP_result['mar_small'])
        self.log('mAR top 1', mAP_result['mar_1'])
        self.val_map.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
        scheduler = CosineAnnealingLR(optimizer, 300, last_epoch=-1)
        return [optimizer], [scheduler]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, help='Path to train directory')
    parser.add_argument('--val_dir', type=str, help='Path to validation directory')
    parser.add_argument('--width', type=int, help='Image width')
    parser.add_argument('--height', type=int, help='Image height')
    parser.add_argument('--backbone', type=str, help='Backbone name')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--obt', action='store_true', help='Flag for one batch test')

    args = parser.parse_args()
    print("Building model...")
    model = get_model(backbone_name = args.backbone, size=(args.width, args.height))
    print("Loading data...")
    dm = DetectionDataModule(args.train_dir, args.val_dir, args.width, args.height, args.batch_size)
    lit_model = LitDetector(model)
    wandb_logger = WandbLogger(project="Ball Detection Table Set")

    checkpoint_callback = ModelCheckpoint(
        save_top_k=10,
        monitor="epoch",
        mode="max",
        dirpath=f"checkpoint/{datetime.now().strftime('%H-%M-%S')}",
        filename="detection-" + args.backbone + "-{epoch:02d}",
    )
    print("Starting training...")
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = L.Trainer(logger=wandb_logger, accelerator="auto", fast_dev_run=args.obt, max_epochs=args.epochs, callbacks=[checkpoint_callback, lr_monitor], log_every_n_steps=10)
    trainer.fit(lit_model, dm)
