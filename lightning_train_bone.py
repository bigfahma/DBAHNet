from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pytorch_lightning as pl
from trainer_bone_dbahnet import DBAHNET
from pytorch_lightning.loggers import TensorBoardLogger
import argparse
import torch


if __name__ =="__main__":
    
    print("Number of GPUs available : ",torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(torch.cuda.get_device_properties(i).name)
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, required=True)
    parser.add_argument('--model', type=str, choices=list(["UNET","ATTENTIONUNET","UNETR","SWINUNETR", "DBAHNET"]),default="DBAHNet", help="Choose a model from the list: [UNET, ATTENTIONUNET, UNETR, SWINUNETR,DBAHNET]") 
    args = parser.parse_args()
    if args.model == "UNET":
         from trainer_bone_unet import BONEUNET
         model = BONEUNET()
    if args.model == "ATTENTIONUNET":
         from trainer_bone_attentionunet import BONEATTENTIONUNET
         model = BONEATTENTIONUNET()
    if args.model == "UNETR":
         from trainer_bone_unetr import BONEUNETR
         model = BONEUNETR()
    if args.model == "SWINUNETR":
         from trainer_bone_swinunetr import BONESWINUNETR
         model = BONESWINUNETR()
    if args.model == "DBAHNET":
         from trainer_bone_dbahnet import DBAHNET
         model = DBAHNET()


    print("Training ...")
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val/MeanDiceScore',
        dirpath='./ckpt/{}'.format(args.exp),
        filename='Epoch{epoch:3d}-MeanDiceScore{val/MeanDiceScore:.4f}',
        save_top_k=3,
        mode='max',
        save_last= True,
        auto_insert_metric_name=False
    )
    early_stop_callback = EarlyStopping(
    monitor='val/MeanDiceScore',
    min_delta=0.0001,
    patience=20,
    verbose=False,
    mode='max'
    )
    tensorboardlogger = TensorBoardLogger(
        'logs',
        name = args.exp,
        default_hp_metric = None
    )
    trainer = pl.Trainer(
                          accelerator='gpu',
                          devices= "auto",
                            precision=16,
                        max_epochs = 100,
                        callbacks=[checkpoint_callback, early_stop_callback, ], 
                        num_sanity_val_steps=4,
                        logger = tensorboardlogger,
                        log_every_n_steps= 10,
                        accumulate_grad_batches= 4,
                        )
    trainer.fit(model) 