

from pathlib import Path
import pickle

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from finetune import load_model
from model.densecap import DenseCapModel
from utils.snare_dataset import SnareDataset

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def print_decode_caption(cap: torch.Tensor, idx_to_token):
    for i in cap:
        if i < 1:
            break
        print(idx_to_token[i.item()], end=" ")
    
    print("\n")


def train(model: DenseCapModel, data_loader: DataLoader, idx_to_token):
    pass


def test(model: DenseCapModel, data_loader: DataLoader, idx_to_token):

    model.eval()    
    view_ids = torch.arange(8)

    n = 0
    pos = 0

    with torch.no_grad():
        for batch in tqdm(data_loader):    
            (key1_imgs, key2_imgs), gt_idx, (key1, key2), annotation, is_visual = batch 

            if not is_visual:
                print("skip non visual")
                continue

            key1_imgs = [k.squeeze().to(device) for k in key1_imgs]
            key2_imgs = [k.squeeze().to(device) for k in key2_imgs]
            annotation = annotation[0]
            
            if gt_idx > 0:
                key1_imgs, key2_imgs = key2_imgs, key1_imgs
                    
            losses1, min_loss_cap1 = model.query_caption(key1_imgs, [annotation], view_ids)        
            losses2, min_loss_cap2 = model.query_caption(key2_imgs, [annotation], view_ids)                

            print(f"gt annot: {annotation}")
            n += 1
            if losses2['caption_min'] > losses1['caption_min']:
                pos += 1
                print("correct!")
                print_decode_caption(min_loss_cap1, idx_to_token)        
            else:
                print_decode_caption(min_loss_cap2, idx_to_token)

            print(f"{pos/n:.2f}")

    print(f"end: {pos/n:.2f}")


def main():
    lut_path = Path("./data/VG-regions-dicts-lite.pkl")

    with open(lut_path, 'rb') as f:
        look_up_tables = pickle.load(f)

    idx_to_token = look_up_tables['idx_to_token']
    token_to_idx = look_up_tables['token_to_idx']

    params_path = Path("compute_model_params")
    model_name = "without_aux"
    model = load_model(
        params_path / model_name / "config.json", 
        params_path / (model_name + ".pth.tar"), 
        return_features=False)
    model.token_to_idx = token_to_idx
    
    model.toDevice(device)
    test_set = SnareDataset(mode="test")
    test_loader = DataLoader(test_set, batch_size=1)

    test(model, test_loader, idx_to_token)


if __name__ == "__main__":
    main()