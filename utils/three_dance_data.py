import os
import re
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image


class CarClassImageDataset(Dataset):
    CLASS_NAMES = ['BACK_LEFT', 'BACK_RIGHT', 'FRONT_LEFT', 'FRONT_RIGHT']
    def __init__(self, img_dir, transform=None):
        self.img_dir = Path(img_dir)
        self.model_names = [p.name for p in os.scandir(img_dir) if p.is_dir()]
        self.model_count = len(self.model_names)

        self.model_image_dict = {
            model: { cls: [p.name for p in os.scandir(self.img_dir / model / cls) if p.is_file()] for cls in self.CLASS_NAMES } for model in self.model_names
        }
        self.img_count = np.sum([len(v) for item in self.model_image_dict.values() for v in item.values() ])
        self.transform = transform
        self.cam_regex = re.compile(r"(-?\d+\.\d{3})")

    def __len__(self):
        return self.model_count

    def __getitem__(self, idx):
        model_name = self.model_names[idx]
        img_cls_paths = [(cls, self.img_dir / model_name / cls / p.name) for cls in self.CLASS_NAMES for p in os.scandir(self.img_dir / model_name / cls) if p.is_file()]
        images = [Image.open(str(img_path)).convert("RGB") for _, img_path in img_cls_paths]
        classes = [cls for cls, _ in img_cls_paths]
        cam_poses = [np.array(self.cam_regex.findall(str(path))).astype(float) for path in img_cls_paths]

        if self.transform:
            images = [self.transform(image) for image in images]
        else:
            images = [transforms.ToTensor()(img) for img in images]

        return images, classes, cam_poses, model_name
