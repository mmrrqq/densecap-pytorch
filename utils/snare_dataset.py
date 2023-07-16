import os
import json
from pathlib import Path
import re
import torch
import torch.utils.data
import torchvision.transforms as transforms

import numpy as np
import json

from PIL import Image


shapenet_base_path = Path("../snare/data/screenshots")
annotations_path = Path("../snare/amt/folds_adversarial")


class SnareDataset(torch.utils.data.Dataset):

    def __init__(self, mode='train', transform=None):
        self.total_views = 14        
        self.mode = mode        
        self.transform = transform
        self.view_id_regex = re.compile(r"\w*-([0-9]+)\.png")

        self.load_entries()

    def load_entries(self):
        train_train_files = ["train.json"]
        train_val_files = ["val.json"]
        test_test_files = ["test.json"]

        # modes
        if self.mode == "train":
            self.files = train_train_files
        elif self.mode  == 'valid':
            self.files = train_val_files
        elif self.mode == "test":
            self.files =  test_test_files
        else:
            raise RuntimeError('mode not recognized, should be train, valid or test: ' + str(self.mode))

        # load amt data
        self.data = []
        for file in self.files:
            fname_rel = annotations_path / file
            print(fname_rel)
            with open(fname_rel, 'r') as f:
                self.data = self.data + json.load(f)

        print(f"Loaded Entries. {self.mode}: {len(self.data)} entries")            

    def __len__(self):
        return len(self.data)    
    
    def get_imgs(self, key: str):
        model_path = shapenet_base_path / key        

        intermediate_dict = {}        
        for p in model_path.iterdir():
            if p.is_file() and "png" in p.name:
                view_id = int(self.view_id_regex.match(p.name).group(1))
                if view_id < 6:  # discard first six canonical views
                    continue

                img = Image.open(p.absolute()).convert("RGB")
                if self.transform is not None:
                    img = self.transform(img)
                else:
                    img = transforms.ToTensor()(img)
                
                intermediate_dict[view_id] = img

        keys = list(intermediate_dict.keys()) 
        
        sort_idx = np.argsort(keys)

        imgs = []
        for i in sort_idx:
            imgs.append(intermediate_dict[keys[i]])
        
        return imgs

    def __getitem__(self, idx):
        entry = self.data[idx]

        # get keys
        entry_idx = entry['ans'] if 'ans' in entry else -1 # test set does not contain answers
        if len(entry['objects']) == 2:
            key1, key2 = entry['objects']

        # fix missing key in pair
        else:
            raise NotImplementedError
            key1 = entry['objects'][entry_idx]
            while True:
                key2 = np.random.choice(list(self.img_feats.keys())).split("-")[0]
                if key2 != key1:
                    break                    

        # annotation
        annotation = entry['annotation']
        is_visual = entry['visual'] if 'ans' in entry else -1 # test set does not have labels for visual and non-visual categories

        # load images for keys
        key1_imgs = self.get_imgs(key1)
        key2_imgs = self.get_imgs(key2)                

        return (
            (key1_imgs, key2_imgs),            
            entry_idx,
            (key1, key2),
            annotation,
            is_visual,
        )