import torch
from monai.metrics import DiceMetric#, compute_meandice
from monai.metrics.hausdorff_distance import compute_hausdorff_distance
from monai.losses import DiceLoss,DiceFocalLoss,FocalLoss, DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete, Activations, Compose, EnsureType
from models.DBAHNet import DBAHNet
import pytorch_lightning as pl
import numpy as np
from bone_data.bone_data import get_train_dataloader, get_val_dataloader, get_test_dataloader

import matplotlib.pyplot as plt
from monai.data import decollate_batch

import csv
import os

class DBAHNET(pl.LightningModule):
    def __init__(self, lr=1e-4):
        super().__init__()
        self.lr = lr
        IMAGE_DIM = (32, 320, 320)    
        NUM_CLASSES = 3
        self.emb_dim = 48
        self.num_heads = [6, 12, 24, 48]
        self.depth = 2
        self.window_size = 7
        
        self.model = DBAHNet(emb_dim= self.emb_dim, in_dim=IMAGE_DIM, num_classes= NUM_CLASSES,
                      depth = self.depth, num_heads=self.num_heads, window_size=self.window_size)
        
        self.custom_loss = DiceCELoss(to_onehot_y=False, softmax=True, squared_pred=True)
        self.training_step_outputs = []
        self.test_step_outputs = []
        self.dice_metric = DiceMetric(include_background=True, reduction="mean_batch",ignore_empty=True)

        ##### metrics tracking #####
        self.val_mean_dice = []
        self.val_dice_cortical = []
        self.val_dice_trabecular = []
        self.val_losses = []
        self.test_mean_dice = []
        self.test_dice_cortical = []
        self.test_dice_trabecular = []
        self.test_hd_distance_trab = []
        self.test_hd_distance_cor = []
        self.test_hd_distance_mean = []
        self.post_trans_images = Compose([EnsureType(), 
        AsDiscrete(argmax=True, to_onehot=NUM_CLASSES),])

        self.best_val_dice = 0
        self.best_val_epoch = 0
        self.post_labels = Compose([EnsureType()])


    def forward(self, x):
        return self.model(x) 
    def training_step(self, batch, batch_index):
        inputs, labels = (batch['image'], batch['label'])
        outputs = self.forward(inputs)
        loss = self.custom_loss(outputs, labels)
        self.training_step_outputs.append(loss)
        self.log('train/loss', loss)
        return loss
    
    def validation_step(self, batch, batch_index):
        inputs, labels = batch['image'], batch['label']
        outputs = sliding_window_inference(inputs,(32, 320, 320)  , 1, self.model, overlap=0.5)
        loss = self.custom_loss(outputs, labels)
        outputs = [self.post_trans_images(val_pred_tensor) for val_pred_tensor in decollate_batch(outputs)]
        labels = [self.post_labels(val_pred_tensor)  for val_pred_tensor in decollate_batch(labels)]
        self.dice_metric(y_pred=outputs, y=labels)
        self.val_losses.append(loss)
        return loss
    def on_validation_epoch_end(self):
        mean_val_loss = torch.stack(self.val_losses).mean()
        dice_scores = self.dice_metric.aggregate()
 
        mean_dice_c1 = dice_scores[0].item()
        mean_dice_c2 = dice_scores[1].item()
        mean_dice_c3 = dice_scores[2].item()
        mean_val_dice = (mean_dice_c2 + mean_dice_c3) / 2
        #print(mean_val_dice, mean_dice_c1, mean_dice_c2, mean_dice_c3)
        self.log('val/val_loss', mean_val_loss, sync_dist= True, on_epoch=True, prog_bar=True)
        self.log('val/MeanDiceScore', mean_val_dice, sync_dist= True, on_epoch=True, prog_bar=True)
        self.log('val/Dice_Class1(BG)', mean_dice_c1, sync_dist= True, on_epoch=True, prog_bar=True)
        self.log('val/Dice_Class2(Cortical)', mean_dice_c2, sync_dist= True, on_epoch=True, prog_bar=True)
        self.log('val/Dice_Class3(Trabecular)', mean_dice_c3, sync_dist= True, on_epoch=True, prog_bar=True)
        
        self.dice_metric.reset()
        self.val_losses.clear() 
        tensorboard_logs = {
            "val_dice": mean_val_dice,
            "val_loss": mean_val_loss,
        }        
        
        if mean_val_dice > self.best_val_dice:
            self.best_val_dice = mean_val_dice
            self.best_val_epoch = self.current_epoch
        print(
                f"\n Current epoch: {self.current_epoch} Current mean dice(BG NI): {mean_val_dice:.4f}"
                f" BG: {mean_dice_c1:.4f}"
                f" Cortical: {mean_dice_c2:.4f} Trabecular: {mean_dice_c3:.4f}"
                f"\n Best mean dice: {self.best_val_dice}"
                f" at epoch: {self.best_val_epoch}"
            )
        return {"log": tensorboard_logs}  
    
    def test_step(self, batch, batch_index):
        inputs, labels = batch['image'], batch['label']
        outputs = sliding_window_inference(inputs,  (32, 320, 320), 1, self.model, overlap=0.5)
        loss = self.custom_loss(outputs, labels)
        outputs = [self.post_trans_images(val_pred_tensor) for val_pred_tensor in decollate_batch(outputs)]
        labels = [self.post_labels(val_pred_tensor)  for val_pred_tensor in decollate_batch(labels)]
        self.dice_metric(y_pred=outputs, y=labels)
        return loss

    def on_test_epoch_end(self):
        dice_scores = self.dice_metric.aggregate()
 
        mean_dice_c1 = dice_scores[0].item()
        mean_dice_c2 = dice_scores[1].item()
        mean_dice_c3 = dice_scores[2].item()
        mean_test_dice = (mean_dice_c1 + mean_dice_c2 + mean_dice_c3) / 3
        print(f'Mean Test Dice score : {mean_test_dice}. \n Test dice score per class : C1 BG : {mean_dice_c1}, C2 Cortical: {mean_dice_c2}, C3 Trabecular: {mean_dice_c3}')
     
        self.dice_metric.reset()
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
                    self.model.parameters(), self.lr, weight_decay=1e-5, amsgrad=True
                    )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 200)
        return [optimizer], [scheduler]
    
    def train_dataloader(self):
        return get_train_dataloader()
    
    def val_dataloader(self):
        return get_val_dataloader()
    
    def test_dataloader(self):
        return get_test_dataloader()
    
   


