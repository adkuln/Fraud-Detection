import pytorch_lightning as pl
import torch.nn as nn
from adamp import AdamP
import torch
from model import ClassificationEmbdNN
import numpy as np


class Simple_Trainer(pl.LightningModule):
    '''
    simple feedforward model for binary classification.
    there are options:
    pos_weight for BCEWithLogitsLoss
    weigted random sampler
    '''
    def __init__(self, 
                 cat_dim,
                 continuous,
                 learning_rate,
                 num_outputs=1,
                 ):
        super(Simple_Trainer, self).__init__()
        if continuous is None:
            pass
        else:
            continuous = len(continuous)
        self.model = ClassificationEmbdNN(emb_dims=cat_dim, no_of_cont=continuous)
        self.criterion = nn.BCEWithLogitsLoss()
        self.learning_rate = learning_rate
        
    def forward(self, x, mask=None):
        cat = x[0].to(x[0].device, dtype=torch.long)
        cont = x[1].to(x[0].device, dtype=torch.float)
        out = self.model(cat, cont)
        return out
    
    def configure_optimizers(self):
        optimizer = AdamP([p for p in self.model.parameters() if p.requires_grad],
                          lr=self.learning_rate, betas=(0.9, 0.999), weight_decay=0.5e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=1,
            min_lr=1e-12,
            verbose=True,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": 'val_loss',
        }

    def training_step(self, batch, batch_idx):
        x, y = batch['data'], batch['target']
        y = y.unsqueeze(1)
        yhat = self(x)
        loss = self.criterion(yhat.float(), y.float())
        self.log(f"train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['data'], batch['target']
        y = y.unsqueeze(1)
        yhat = self(x)
        loss = self.criterion(yhat.float(), y.float())
        self.log(f"val_loss", loss, prog_bar=True, on_epoch=True)
        return loss