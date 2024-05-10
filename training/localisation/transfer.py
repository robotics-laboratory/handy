import lightning as L
import torch
import torch.nn as nn
import argparse


from torch.optim.lr_scheduler import CosineAnnealingLR
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from torchmetrics.classification import BinaryPrecision, BinaryRecall
from datetime import datetime
from model import TTNetWithProb, BallLocalisation
from datasets import ClassificationDataModule


class LitClassification(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

        for param in self.model.backbone.parameters():
            param.requires_grad = False
        
        for param in self.model.head.parameters():
            param.requires_grad = False
        
        self.loss = nn.CrossEntropyLoss()
        self.pr = BinaryPrecision()
        self.re = BinaryRecall()

    def training_step(self, batch, batch_idx):
        images, cls = batch
        _, pred_logits = self.model(images)
        loss = self.loss(pred_logits, cls)

        pred_cls = torch.argmax(pred_logits, dim=1)
        self.pr.update(pred_cls, cls)
        self.re.update(pred_cls, cls)

        self.log('train_loss', loss, on_step=True, on_epoch=True)

        return loss

    def on_train_epoch_end(self):
        self.log('train_precision', self.pr.compute())
        self.log('train_recall', self.re.compute())
        self.pr.reset()
        self.re.reset()
    
    def validation_step(self, batch, batch_idx):
        images, cls = batch
        _, pred_logits = self.model(images)
        loss = self.loss(pred_logits, cls)

        pred_cls = torch.argmax(pred_logits, dim=1)
        self.pr.update(pred_cls, cls)
        self.re.update(pred_cls, cls)

        self.log('val_loss', loss, on_step=True, on_epoch=True)

        return loss

    def on_validation_epoch_end(self):
        self.log('val_precision', self.pr.compute())
        self.log('val_recall', self.re.compute())
        self.pr.reset()
        self.re.reset()
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
        scheduler = CosineAnnealingLR(optimizer, 15, last_epoch=-1)
        return [optimizer], [scheduler]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the localisation model')
    parser.add_argument('--train_dir', type=str, help='Path to train directory')
    parser.add_argument('--val_dir', type=str, help='Path to validation directory')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--n_last', type=int, default=5, help='Number of last frames to consider')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--dropout_p', type=float, default=0, help='Dropout probability')
    parser.add_argument('--checkpoint', type=str, help='Path to the checkpoint')
    parser.add_argument('--width', type=int, default=320, help='Width of the input image')
    parser.add_argument('--height', type=int, default=192, help='Height of the input image')
    args = parser.parse_args()



    upst_model = BallLocalisation(dropout_p=args.dropout_p, n_last=args.n_last)
    checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    upst_model.load_state_dict({k[6:]: v for k, v in checkpoint["state_dict"].items() if k.startswith("model.")})
    model = TTNetWithProb(upst_model)  

    data_module = ClassificationDataModule(args.train_dir, args.val_dir, args.width, args.height, args.n_last, args.batch_size)
    lit_model = LitClassification(model)

    wandb_logger = WandbLogger(project='localisation-classification')
    checkpoint_callback = ModelCheckpoint(
        save_top_k=10,
        monitor="epoch",
        mode="max",
        dirpath=f"checkpoint/{datetime.now().strftime('%m-%d-%H-%M-%S')}",
        filename="localisation-class-{epoch:02d}",
    )
    trainer = L.Trainer(max_epochs=args.epochs, logger=wandb_logger, callbacks=[checkpoint_callback], log_every_n_steps=10)
    trainer.fit(lit_model, data_module)
    