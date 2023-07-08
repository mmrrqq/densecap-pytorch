import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from dataset import DenseCapDataset
from evaluate import quantity_check
from model.densecap import densecap_resnet50_fpn
from train import save_model
from utils.filtered_car_data_loader import FilteredCarClassImageDataset

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_DIR_ROOT = '../3-dance/datasets/vg'
VG_DATA_PATH = './data/VG-regions-lite.h5'
LOOK_UP_TABLES_PATH = './data/VG-regions-dicts-lite.pkl'
BATCH_SIZE = 8
CAP_LR = 1e-3
LR = 1e-4
WEIGHT_DECAY = 0


def load_model(model_config_path: Path, checkpoint_path: Path, return_features=False, box_per_img=50, verbose=False):
    with open(model_config_path, 'r') as f:
        model_args = json.load(f)

    model = densecap_resnet50_fpn(backbone_pretrained=model_args['backbone_pretrained'],
                                  return_features=return_features,
                                  feat_size=model_args['feat_size'],
                                  hidden_size=model_args['hidden_size'],
                                  max_len=model_args['max_len'],
                                  emb_size=model_args['emb_size'],
                                  rnn_num_layers=model_args['rnn_num_layers'],
                                  vocab_size=model_args['vocab_size'],
                                  fusion_type=model_args['fusion_type'],
                                  box_detections_per_img=box_per_img)

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])

    if verbose and 'results_on_val' in checkpoint.keys():
        print('[INFO]: checkpoint {} loaded'.format(checkpoint_path))
        print('[INFO]: correspond performance on val set:')
        for k, v in checkpoint['results_on_val'].items():
            if not isinstance(v, dict):
                print('        {}: {:.3f}'.format(k, v))

    return model


def main():
    params_path = Path("model_params")
    model_name = "without_aux"
    model = load_model(
        params_path / model_name / "config.json", 
        params_path / (model_name + ".pth.tar"), 
        return_features=False)

    val_set = DenseCapDataset(IMG_DIR_ROOT, VG_DATA_PATH, LOOK_UP_TABLES_PATH, dataset_type='val')
    idx_to_token = val_set.look_up_tables['idx_to_token']
    dataset = FilteredCarClassImageDataset("filtered_car_data.pkl")
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=DenseCapDataset.collate_fn)

    model.to(device)
    optimizer = torch.optim.Adam([{'params': (para for name, para in model.named_parameters()
                                                if para.requires_grad and 'box_describer' not in name)},
                                    {'params': (para for para in model.roi_heads.box_describer.parameters()
                                                if para.requires_grad), 'lr': CAP_LR}],
                                    lr=LR, weight_decay=WEIGHT_DECAY)

    iter_counter = 0
    best_map = 0.
    writer = SummaryWriter()

    for epoch in range(10):
        for img, targets in tqdm(data_loader):
            img = [img_tensor.to(device) for img_tensor in img]
            targets = [{k: v.to(device) for k, v in target.items()} for target in targets]

            model.train()

            losses = model(img, targets)

            detect_loss = losses['loss_objectness'] + losses['loss_rpn_box_reg'] + \
                            losses['loss_classifier'] + losses['loss_box_reg']
            caption_loss = losses['loss_caption']

            total_loss = 1.0 * detect_loss + 1.0 * caption_loss        

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            writer.add_scalar('batch_loss/total', total_loss.item(), iter_counter)
            writer.add_scalar('batch_loss/detect_loss', detect_loss.item(), iter_counter)
            writer.add_scalar('batch_loss/caption_loss', caption_loss.item(), iter_counter)

            writer.add_scalar('details/loss_objectness', losses['loss_objectness'].item(), iter_counter)
            writer.add_scalar('details/loss_rpn_box_reg', losses['loss_rpn_box_reg'].item(), iter_counter)
            writer.add_scalar('details/loss_classifier', losses['loss_classifier'].item(), iter_counter)
            writer.add_scalar('details/loss_box_reg', losses['loss_box_reg'].item(), iter_counter)

            if iter_counter > 0 and iter_counter % 400 == 0:
                try:
                    results = quantity_check(model, val_set, idx_to_token, device, verbose=False)
                    if results['map'] > best_map:
                        best_map = results['map']
                        save_model(model, optimizer, None, results, iter_counter)
                    
                    writer.add_scalar('metric/map', results['map'], iter_counter)
                    writer.add_scalar('metric/det_map', results['detmap'], iter_counter)

                except AssertionError as e:
                    print('[INFO]: evaluation failed at epoch {}'.format(epoch))
                    print(e)

            iter_counter += 1

    save_model(model, optimizer, None, results, iter_counter, flag='end')    
    writer.close()

if __name__ == "__main__":
    main()