# DBAHNet
DBAHNET: DUAL-BRANCH ATTENTION-BASED HYBRID NETWORK FOR HIGH-RESOLUTION 3D MICRO-CT BONE SCAN SEGMENTATION (ISBI 2024 Submission - Pending)

# Python libraries versions
You can install the python libraries using the pip install command.
torch : 1.13.1+cu117
monai : 1.2.0
pytorch lightning : 2.0.7
SimpleITK
numpy
json
pyvista (for visualisation)
csv

All training and experiments were conducted with 4 NVIDIA V100 32GB GPUs.

# Dataset
The JSON format organizes a dataset into three subsets: "train," "val," and "test." Each subset comprises dictionaries representing image-label pairs.

![image](https://github.com/bigfahma/DBAHNet/assets/85291758/9b0cb065-8fb4-478e-a0dc-6240bdf98f6b)

# Train DBAHNet 
python lightning_train_bone.py --model  name_model --exp name_of_experience 

where :

name_model is the name of the model to train. Can be either one of : ["UNET","ATTENTIONUNET","UNETR","SWINUNETR", "DBAHNET"].
name_of_experience is the name chosen for the training experiment.

# Test DBAHNet

python lightning_bone_test.py --model name_model --ckpt "ckpt/name_of_experience/last.ckpt"

where :
name_model is the name of the model to test. Can be either one of : ["UNET","ATTENTIONUNET","UNETR","SWINUNETR", "DBAHNET"].
name_of_experience is the name chosen for the training experiment. ( -- ckpt Path of the checkpoint of the last trained model .ckpt)

