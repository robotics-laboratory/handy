import lightning as L
import torch
import torch.nn as nn
import argparse


from torch.optim.lr_scheduler import StepLR
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from model import BallLocalisation
from datasets import LocalisationDataModule


def gaussian_1d(pos, muy, sigma):
    """Create 1D Gaussian distribution based on ball position (muy), and std (sigma)"""
    target = torch.exp(- (((pos - muy) / sigma) ** 2) / 2)
    return target

def normal_dist_target(center, rad, width, height, sigma, device='cpu'):
    w, h = width, height
    if sigma is None:
        sigma = rad
    target_ball_position = torch.zeros((w + h,), device=device)
    # Only do the next step if the ball is existed
    if (w > center[0] > 0) and (h > center[1] > 0):
        # For x
        x_pos = torch.arange(0, w, device=device)
        target_ball_position[:w] = gaussian_1d(x_pos, center[0], sigma=sigma)
        # For y
        y_pos = torch.arange(0, h, device=device)
        target_ball_position[w:] = gaussian_1d(y_pos, center[1], sigma=sigma)

        target_ball_position[target_ball_position < 0.05] = 0.

    return target_ball_position

class Ball_Detection_Loss(nn.Module):
    def __init__(self, w, h, epsilon=1e-9):
        super(Ball_Detection_Loss, self).__init__()
        self.w = w
        self.h = h
        self.epsilon = epsilon

    def forward(self, pred_ball_position, target_ball_position):
        x_pred = pred_ball_position[:, :self.w]
        y_pred = pred_ball_position[:, self.w:]

        x_target = target_ball_position[:, :self.w]
        y_target = target_ball_position[:, self.w:]

        loss_ball_x = - torch.mean(x_target * torch.log(x_pred + self.epsilon) + (1 - x_target) * torch.log(1 - x_pred + self.epsilon))
        loss_ball_y = - torch.mean(y_target * torch.log(y_pred + self.epsilon) + (1 - y_target) * torch.log(1 - y_pred + self.epsilon))

        return loss_ball_x + loss_ball_y

def get_predicted_ball_pos(prob, width, thesh=0.0001):
    pred = prob.clone()
    pred[pred < thesh] = 0
    pred_x = torch.argmax(pred[:, :width], dim=1)
    pred_y = torch.argmax(pred[:, width:], dim=1)
    return torch.stack((pred_x, pred_y), dim=1).to(pred.device)


class LitLocalisation(L.LightningModule):
    def __init__(self, model, sigma=None):
        super().__init__()
        self.model = model
        self.sigma = sigma
        self.loss = Ball_Detection_Loss(model.width, model.height)

    def training_step(self, batch, batch_idx):
        images, data = batch
        pred = self.model(images)

        target = torch.zeros_like(pred, device=self.device)
        rad = torch.zeros((pred.size(0), 1), device=self.device)
        ball_coords = torch.zeros((pred.size(0), 2), device=self.device)
        for sample_index in range(pred.size(0)):
            target[sample_index] = normal_dist_target(data[sample_index]['ball_center'], data[sample_index]['ball_rad'],
                                                    self.model.width, self.model.height, self.sigma, self.device)
            rad[sample_index] = data[sample_index]['ball_rad']
            ball_coords[sample_index] = torch.tensor(data[sample_index]['ball_center'], device=self.device)
        
        loss = self.loss(pred, target)
        pred_coord = get_predicted_ball_pos(pred, self.model.width)
        dist = torch.sqrt(torch.sum((pred_coord - ball_coords) ** 2, dim=1))
        relative_dist = torch.mean(dist / rad)
        rmse = torch.sqrt(torch.mean(dist**2))
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        self.log('train_relative_dist', relative_dist, on_step=True, on_epoch=True)
        self.log('train_rmse', rmse, on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, data = batch
        pred = self.model(images)

        target = torch.zeros_like(pred, device=self.device)
        rad = torch.zeros((pred.size(0), 1), device=self.device)
        ball_coords = torch.zeros((pred.size(0), 2), device=self.device)
        for sample_index in range(pred.size(0)):
            target[sample_index] = normal_dist_target(data[sample_index]['ball_center'], data[sample_index]['ball_rad'],
                                                    self.model.width, self.model.height, self.sigma, self.device)
            rad[sample_index] = data[sample_index]['ball_rad']
            ball_coords[sample_index] = torch.tensor(data[sample_index]['ball_center'], device=self.device)
        
        loss = self.loss(pred, target)
        pred_coord = get_predicted_ball_pos(pred, self.model.width)
        dist = torch.sqrt(torch.sum((pred_coord - ball_coords) ** 2, dim=1))
        relative_dist = torch.mean(dist / rad)
        rmse = torch.sqrt(torch.mean(dist**2))
        self.log('val_loss', loss, on_step=True, on_epoch=True)
        self.log('val_relative_dist', relative_dist, on_step=True, on_epoch=True)
        self.log('val_rmse', rmse, on_step=True, on_epoch=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [scheduler]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the localisation model')
    parser.add_argument('--data_dir', type=str, help='Path to the data directory')
    parser.add_argument('--annot_file', type=str, help='Path to the annotation file')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--width', type=int, default=320, help='Width of the input image')
    parser.add_argument('--height', type=int, default=192, help='Height of the input image')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--dropout_p', type=float, default=0.5, help='Dropout probability')
    parser.add_argument('--sigma', type=float, default=None, help='Sigma for the Gaussian distribution')
    args = parser.parse_args()

    data_dir = args.data_dir
    annot_file = args.annot_file
    width = args.width
    height = args.height
    batch_size = args.batch_size
    sigma = args.sigma
    model = BallLocalisation(dropout_p=args.dropout_p)
    data_module = LocalisationDataModule(data_dir, annot_file, width, height, batch_size)
    lit_model = LitLocalisation(model, sigma)

    wandb_logger = WandbLogger(project='localisation')
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', dirpath='checkpoints', filename='localisation-{epoch:02d}-{val_loss:.2f}')

    trainer = L.Trainer(max_epochs=args.epochs, logger=wandb_logger, callbacks=[checkpoint_callback], log_every_n_steps=10)
    trainer.fit(lit_model, data_module)
    