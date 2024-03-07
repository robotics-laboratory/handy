import lightning as L
import torch
import argparse
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.optim.lr_scheduler import MultiStepLR
from pytorch_lightning.loggers import WandbLogger

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
        self.log('mAP_medium', mAP_result['map_medium'])
        self.log('mAP@0.50', mAP_result['map_50'])
        self.log('mAP@0.50:0.95', mAP_result['map'])
        self.val_map.reset()
        
        scheduler = self.lr_schedulers()
        scheduler.step()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.0005, momentum=0.9, nesterov=True)
        scheduler = MultiStepLR(optimizer=optimizer, milestones=[45], gamma=0.1, verbose=True) 
        return [optimizer], [scheduler]
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='Path to directory with images')
    parser.add_argument('--annot_file', type=str, help='Path to annotation file')
    parser.add_argument('--width', type=int, help='Image width')
    parser.add_argument('--height', type=int, help='Image height')
    parser.add_argument('--backbone', type=str, help='Backbone name')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--obt', action='store_true', help='Flag for one batch test')

    args = parser.parse_args()

    model = get_model(backbone_name = args.backbone, size=args.width)
    dm = DetectionDataModule(args.data_dir, args.annot_file, args.width, args.height, args.batch_size)
    lit_model = LitDetector(model)
    wandb_logger = WandbLogger(project="Ball Deection")
    trainer = L.Trainer(logger=wandb_logger, accelerator="auto", fast_dev_run=args.obt, max_epochs=args.epochs)
    trainer.fit(lit_model, dm)
