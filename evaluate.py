import argparse
from pathlib import Path
import pickle
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.data_loader import DenseCapDataset
from model.evaluator import DenseCapEvaluator
from utils.model_io import load_model


def quality_check(model, dataset, idx_to_token, device, max_iter=-1):

    model.to(device)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1,
                                 pin_memory=True, collate_fn=DenseCapDataset.collate_fn)

    print('[quality check]')
    for i, (img, targets, info) in enumerate(data_loader):

        img = [img_tensor.to(device) for img_tensor in img]
        targets = [{k: v.to(device) for k, v in target.items()} for target in targets]

        with torch.no_grad():
            model.eval()
            model.return_features = False
            detections = model(img)

        for j in range(len(targets)):
            print('<{}>'.format(info[j]['file_name']))
            print('=== ground truth ===')
            for box, cap, cap_len in zip(targets[j]['boxes'], targets[j]['caps'], targets[j]['caps_len']):
                print('box:', box.tolist())
                print('len:', cap_len.item())
                print('cap:', ' '.join(idx_to_token[idx] for idx in cap.tolist() if idx_to_token[idx] != '<pad>'))
                print('-'*20)

            print('=== predict ===')
            for box, cap, score in zip(detections[j]['boxes'], detections[j]['caps'], detections[j]['scores']):
                print('box:', [round(c, 2) for c in box.tolist()])
                print('score:', round(score.item(), 2))
                print('cap:', ' '.join(idx_to_token[idx] for idx in cap.tolist() if idx_to_token[idx] != '<pad>'))
                print('-'*20)

        if i >= max_iter > 0:
            break


def quantity_check(model, dataset, idx_to_token, device, max_iter=-1, verbose=True):

    model.to(device)
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2,
                                 pin_memory=True, collate_fn=DenseCapDataset.collate_fn)

    evaluator = DenseCapEvaluator(list(model.roi_heads.box_describer.special_idx.keys()))

    print('[quantity check]')
    for i, (img, targets, info) in tqdm(enumerate(data_loader), total=len(data_loader)):

        img = [img_tensor.to(device) for img_tensor in img]
        targets = [{k: v.to(device) for k, v in target.items()} for target in targets]

        with torch.no_grad():
            model.eval()
            model.return_features = False
            detections = model(img)

        for j in range(len(targets)):
            scores = detections[j]['scores']
            boxes = detections[j]['boxes']
            text = [' '.join(idx_to_token[idx] for idx in cap.tolist() if idx_to_token[idx] != '<pad>')
                    for cap in detections[j]['caps']]
            target_boxes = targets[j]['boxes']
            target_text = [' '.join(idx_to_token[idx] for idx in cap.tolist() if idx_to_token[idx] != '<pad>')
                    for cap in targets[j]['caps']]
            img_id = info[j]['file_name']

            evaluator.add_result(scores, boxes, text, target_boxes, target_text, img_id)

        if i >= max_iter > 0:
            break

    results = evaluator.evaluate(verbose)
    # if verbose:
    print('MAP: {:.3f} DET_MAP: {:.3f}'.format(results['map'], results['detmap']))

    return results


def parse_args():    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--params-path", default="model_params")
    parser.add_argument("--model-name", default="gpuX2")
    parser.add_argument("--data-path", default="data")

    args = parser.parse_args()
    args.data_path = Path(args.data_path)

    return args


def main(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")    
    val_set = DenseCapDataset(args.data_path / "vg", args.data_path / "VG-regions-lite.h5", args.data_path / "VG-regions-dicts-lite.pkl", dataset_type='val')
    idx_to_token = val_set.look_up_tables['idx_to_token']

    params_path = Path(args.params_path)
    model = load_model(
        params_path / "config.json",
        params_path / (args.model_name + ".pth.tar"),
        return_features=False,        
    )
    model.eval()    

    quantity_check(model, val_set, idx_to_token, device)


if __name__ == "__main__":
    args = parse_args()
    main(args)