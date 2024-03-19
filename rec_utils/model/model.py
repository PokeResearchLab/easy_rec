import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

class YourModel(pl.LightningModule):
    def __init__(self, cfg_data):
        self.model = cfg_data['model']

    def forward(self, x):
        # Define the forward pass of your model
        
        return self.model(x)    #MAKES SENSE?????????

    def training_step(self, batch, batch_idx):
        # Define the training step
        x, y = batch
        y_hat = self.forward(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # Define the validation step
        x, y = batch
        y_hat = self.forward(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        # Define the optimizer
        optimizer = optim.Adam(self.parameters(), lr=cfg_data['lr'])
        return optimizer
