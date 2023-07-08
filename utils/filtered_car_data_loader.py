import pickle
import re
from pathlib import Path

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image


class FilteredCarClassImageDataset(Dataset):    
    def __init__(self, path, transform=None):
        self.data_file_path = Path(path)

        with open(self.data_file_path, "rb") as file:            
            self.img_info = pickle.load(file)            
        
        self.keys = list(self.img_info.keys())
        self.transform = transform
        self.cam_regex = re.compile(r"(-?\d+\.\d{3})")

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        img_path = self.keys[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        regions = self.img_info[img_path]
        
        boxes = torch.stack([torch.tensor(region['box'], dtype=torch.float32) for region in regions])
        caps = torch.stack([torch.tensor(region['cap'], dtype=torch.long) for region in regions])
        caps_len = torch.tensor([(cap != 0).sum() for cap in caps], dtype=torch.long)

        targets = {
            'boxes': boxes,
            'caps': caps,
            'caps_len': caps_len
        }


        return img, targets
