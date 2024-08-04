from monai.utils import set_determinism
from monai.transforms import (
    Compose,
    LoadImaged,
    RandFlipd,
    NormalizeIntensityd,
    RandScaleIntensityd,
    RandCropByPosNegLabeld,
    ToTensord,
    AddChanneld,
    MapTransform,
    RandAdjustContrastd,
    Rand3DElasticd,
    RandAffined,
    RandZoomd,
    RandGaussianNoised


)
import torch   
from monai.data import DataLoader, Dataset
import json
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv

def load_data(file_path):
    with open(file_path) as file:
        data = json.load(file)
        return data['train'], data['val']#, data['test']


set_determinism(seed=1)

train_files, val_files = load_data('bone_data/output_data_control.json')#output_data_all.json') # data_bone_08_15_split.json

class PrintShape:
    def __call__(self, data):
        print(f"Image shape: {data['image'].shape}")
        print(f"Label shape: {data['label'].shape}")
        return data

class DataViewImage:
    def __call__(self,data):
        print(np.array(data["image"][0]).shape)
        return data
class DataViewLabel:
    def __call__(self,data):
        pv.plot(np.array(data["label"][0]))
        pv.plot(np.array(data["label"][1]))
        pv.plot(np.array(data["label"][2]))
        return data

class RemapLabels(MapTransform):
    def __init__(self, keys, map_dict):
        super().__init__(keys)
        self.map_dict = map_dict

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.map_dict.get(d[key], d[key])
        return d

class ConvertToMultiChannelForBoneClassesd(MapTransform):

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            result.append(d[key] == 0)
            result.append(d[key] == 1)
            result.append(d[key] == 2)
            multi_channel_label = np.stack(result, axis=0).astype(np.float32)
            d[key] = multi_channel_label
            
        return d
class ReorderDims(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)
    
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = np.transpose(d[key], (2, 0, 1))  # Reorder from (H, W, D) to (D, H, W)
        return d        

class PadToMatchSize(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        key_1, key_2 = self.keys
        if d[key_1].shape != d[key_2].shape:
            max_shape = tuple(max(s1, s2) for s1, s2 in zip(d[key_1].shape, d[key_2].shape))
            for key in self.keys:
                pad_width = [(0, max_dim - curr_dim) for curr_dim, max_dim in zip(d[key].shape, max_shape)]
                d[key] = np.pad(d[key], pad_width, mode='constant', constant_values=0)
        return d

class PadToMinimumSize(MapTransform):
    def __init__(self, keys, min_size):
        super().__init__(keys)
        self.min_size = min_size

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            pad_width = [(0, max(0, min_dim - curr_dim)) for curr_dim, min_dim in zip(d[key].shape, self.min_size)]
            d[key] = np.pad(d[key], pad_width, mode='constant', constant_values=0)
        return d
def get_train_dataloader():

    train_transform = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            ReorderDims(keys=["image", "label"]),
            #PrintShape(),  # Add PrintShape to check dimensions after loading
            PadToMatchSize(keys=["image", "label"]),
            PadToMinimumSize(keys=["image", "label"], min_size=(32, 320, 320)),
            ConvertToMultiChannelForBoneClassesd(keys = ['label']),
            AddChanneld(keys = ["image"]),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(32, 320, 320),
                pos = 0.6),             
            Rand3DElasticd(
                keys=["image", "label"],
                sigma_range=(9, 13),
                magnitude_range=(0, 900),
                prob=0.2,
                rotate_range=(0, 0, 0), 
                shear_range=None,
                translate_range=None,
                scale_range=None,
                mode=('bilinear', 'nearest'),
                padding_mode='border'
            ),
            RandAffined(
                keys=["image", "label"],
                prob=0.2,
                rotate_range=(0, 0, 0),
                scale_range=(0.85, 1.25),
                mode=('bilinear', 'nearest')
            ),  
            RandGaussianNoised(
                keys = ['image'],
                mean = 0.0,
                sigma = 0.1,
            )         
            RandFlipd(keys = ["image", "label"],
                      prob = 0.5,
                      spatial_axis = 0),
             RandFlipd(keys = ["image", "label"],
                      prob = 0.5,
                      spatial_axis = 1),
            RandAdjustContrastd(
                keys = ["image"],
                gamma = (0.5, 4.5),
            ),
            RandScaleIntensityd(keys = "image", prob = 1, factors = 0.1),
            NormalizeIntensityd(keys = "image",
                                nonzero = True,
                                channel_wise = True),
            ToTensord(keys=["image", "label"]),
        ]
    )
    train_ds = Dataset(data=train_files, transform=train_transform)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers = 4, pin_memory = True, )


    return train_loader

def get_val_dataloader():
    val_transform = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            ReorderDims(keys=["image", "label"]),
            PadToMatchSize(keys=["image", "label"]),
            PadToMinimumSize(keys=["image", "label"], min_size=(32, 320, 320)),
            ConvertToMultiChannelForBoneClassesd(keys = ['label']),
            AddChanneld(keys = ["image"]),
            NormalizeIntensityd(keys = "image",
                               nonzero = True,
                               channel_wise = True),
            ToTensord(keys=["image", "label"]),
        ]
    )
    val_ds = Dataset(data=val_files, transform=val_transform)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers = 0, pin_memory = True,)

    return val_loader

def get_test_dataloader():
    test_transform = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            ReorderDims(keys=["image", "label"]),
            ConvertToMultiChannelForBoneClassesd(keys = ['label']),
            AddChanneld(keys = ["image"]),
            NormalizeIntensityd(keys = "image",
                               nonzero = True,
                               channel_wise = True),
            ToTensord(keys=["image", "label"]),
        ]
    )
    test_ds = Dataset(data=test_files, transform=test_transform)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    return test_loader

def inspect_data(dataloader, data_type='train'):
    print(f"Inspecting {data_type} data")
    
    for batch_idx, batch in enumerate(dataloader):
        images, labels = batch['image'], batch['label']
        
        # Print shapes of images and labels
        print(f"Batch {batch_idx+1}")
        print(f"  Images shape: {images.shape}")
        print(f"  Labels shape: {labels.shape}")
        
        # Print unique values in labels
        unique_values = torch.unique(labels)
        print(f"  Unique label values: {unique_values.numpy()}")

if __name__ =='__main__':
    # Create the DataLoaders
    train_loader = get_train_dataloader()
    val_loader = get_val_dataloader()
    test_loader = get_test_dataloader()

    # Inspect data from each DataLoader
    inspect_data(train_loader, data_type='train')
    inspect_data(val_loader, data_type='val')
    inspect_data(test_loader, data_type='test')