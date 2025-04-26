import lightning as L
import torch
import argparse
from datetime import datetime
from torchmetrics.classification import BinaryJaccardIndex
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor


from models import UNet
from dataset import SegmentationDataModule
from loss import DiceLoss

class LitSegmentation(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.dice_loss = DiceLoss()
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        self.train_iou = BinaryJaccardIndex()
        self.val_iou = BinaryJaccardIndex()
    
    def training_step(self, batch, batch_idx):
        images, masks = batch
        pred_mask = self.model(images).squeeze(1)
        bce_loss = self.bce_loss(pred_mask, masks.float())
        dice_loss = self.dice_loss(pred_mask, masks)
        loss = bce_loss + dice_loss

        self.train_iou.update(pred_mask, masks)
        self.log('train_loss', loss.item(), on_step=True, on_epoch=True, batch_size=len(images))
        return loss
    
    def on_train_epoch_end(self):
        iou = self.train_iou.compute()
        self.log('train_iou', iou)
        self.train_iou.reset()

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        pred_mask = self.model(images).squeeze(1)
        bce_loss = self.bce_loss(pred_mask, masks.float())
        dice_loss = self.dice_loss(pred_mask, masks)
        loss = bce_loss + dice_loss

        self.val_iou.update(pred_mask, masks)
        self.log('val_loss', loss.item(), on_step=True, on_epoch=True, batch_size=len(images))
        return loss
    
    def on_validation_epoch_end(self):
        iou = self.val_iou.compute()
        self.log('val_iou', iou)
        self.val_iou.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
        return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5),
                        "monitor": "val_iou",
                        'interval': 'epoch',
                        'frequency': 1,
                        'name': "lr"
                    },
                }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, help='Path to train directory')
    parser.add_argument('--val_dir', type=str, help='Path to validation directory')
    parser.add_argument('--width', type=int, default=256, help='Image width')
    parser.add_argument('--height', type=int, default=256, help='Image height')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--epochs', type=int, help='Number of epochs')

    args = parser.parse_args()

    model = UNet(3, 1, 16)
    dm = SegmentationDataModule(args.train_dir, args.val_dir, size=(args.width, args.height), batch_size=args.batch_size)
    lit_model = LitSegmentation(model)
    wandb_logger = WandbLogger(project="Ball-Segmentation")

    checkpoint_callback = ModelCheckpoint(
        save_top_k=10,
        monitor="epoch",
        mode="max",
        dirpath=f"checkpoint/{datetime.now().strftime('%m-%d-%H-%M-%S')}",
        filename="segmentation-{epoch:02d}",
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = L.Trainer(logger=wandb_logger, accelerator="auto", max_epochs=args.epochs, callbacks=[checkpoint_callback, lr_monitor], log_every_n_steps=10)
    trainer.fit(lit_model, dm)
