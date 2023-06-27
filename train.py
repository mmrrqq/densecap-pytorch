import os
import json

import torch
import numpy as np
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from torchvision.models._api import get_weight
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as f

from utils.data_loader import DenseCapDataset
from model.densecap import densecap_resnet50_fpn

from evaluate import quality_check, quantity_check
from utils.three_dance_data import CarClassImageDataset

torch.backends.cudnn.benchmark = True
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_EPOCHS = 10
USE_TB = True
CONFIG_PATH = './model_params'
MODEL_NAME = 'train_all_val_all_bz_2_epoch_10_inject_init'
IMG_DIR_ROOT = '/home/markus/dev/grit_socket/datasets/vg'
VG_DATA_PATH = './data/VG-regions-lite.h5'
LOOK_UP_TABLES_PATH = './data/VG-regions-dicts-lite.pkl'
MAX_TRAIN_IMAGE = -1  # if -1, use all images in train set
MAX_VAL_IMAGE = -1

CLASSES = {0: 'BACK_LEFT', 1: 'BACK_RIGHT', 2: 'FRONT_LEFT', 3: 'FRONT_RIGHT'}
CLASS_MAPPING = {v: k for k, v in CLASSES.items()}


def set_args():
    args = dict()

    args['backbone_pretrained'] = True
    args['return_features'] = False

    # Caption parameters
    args['feat_size'] = 4096
    args['hidden_size'] = 512
    args['max_len'] = 16
    args['emb_size'] = 512
    args['rnn_num_layers'] = 1
    args['vocab_size'] = 10629
    args['fusion_type'] = 'init_inject'

    # Training Settings
    args['detect_loss_weight'] = 1.
    args['caption_loss_weight'] = 1.
    args['lr'] = 1e-4
    args['caption_lr'] = 1e-3
    args['weight_decay'] = 0.
    args['batch_size'] = 4
    args['use_pretrain_fasterrcnn'] = True
    args['box_detections_per_img'] = 50

    if not os.path.exists(os.path.join(CONFIG_PATH, MODEL_NAME)):
        os.mkdir(os.path.join(CONFIG_PATH, MODEL_NAME))
    with open(os.path.join(CONFIG_PATH, MODEL_NAME, 'config.json'), 'w') as f:
        json.dump(args, f, indent=2)

    return args


def save_model(model, optimizer, scaler, results_on_val, iter_counter, flag=None):
    state = {'model': model.state_dict(),
             'optimizer': optimizer.state_dict(),
             'scaler': scaler.state_dict(),
             'results_on_val': results_on_val,
             'iterations': iter_counter}
    if isinstance(flag, str):
        filename = os.path.join('model_params', '{}_{}.pth.tar'.format(MODEL_NAME, flag))
    else:
        filename = os.path.join('model_params', '{}.pth.tar'.format(MODEL_NAME))
    print('Saving checkpoint to {}'.format(filename))
    torch.save(state, filename)


def train(args):
    print('Model {} start training...'.format(MODEL_NAME))

    model = densecap_resnet50_fpn(backbone_pretrained=args['backbone_pretrained'],
                                  feat_size=args['feat_size'],
                                  hidden_size=args['hidden_size'],
                                  max_len=args['max_len'],
                                  emb_size=args['emb_size'],
                                  rnn_num_layers=args['rnn_num_layers'],
                                  vocab_size=args['vocab_size'],
                                  fusion_type=args['fusion_type'],
                                  box_detections_per_img=args['box_detections_per_img'])
    if args['use_pretrain_fasterrcnn']:
        weights = get_weight('FasterRCNN_ResNet50_FPN_Weights.DEFAULT')
        model.backbone.load_state_dict(fasterrcnn_resnet50_fpn(weights=weights).backbone.state_dict(), strict=False)
        model.rpn.load_state_dict(fasterrcnn_resnet50_fpn(weights=weights).rpn.state_dict(), strict=False)

    model.to(device)

    optimizer = torch.optim.Adam([{'params': (para for name, para in model.named_parameters()
                                              if para.requires_grad and 'box_describer' not in name)},
                                  {'params': (para for para in model.roi_heads.box_describer.parameters()
                                              if para.requires_grad), 'lr': args['caption_lr']}],
                                 lr=args['lr'], weight_decay=args['weight_decay'])

    # apex initialization
    # model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)
    # ref: https://github.com/NVIDIA/apex/issues/441
    # model.roi_heads.box_roi_pool.forward = \
    #     amp.half_function(model.roi_heads.box_roi_pool.forward)

    train_set = DenseCapDataset(IMG_DIR_ROOT, VG_DATA_PATH, LOOK_UP_TABLES_PATH, dataset_type='train')
    val_set = DenseCapDataset(IMG_DIR_ROOT, VG_DATA_PATH, LOOK_UP_TABLES_PATH, dataset_type='val')
    idx_to_token = train_set.look_up_tables['idx_to_token']

    car_dataset = CarClassImageDataset(img_dir='../grit_socket/data/car_images_model_sort')
    car_data_loader = DataLoader(car_dataset, batch_size=1)

    if MAX_TRAIN_IMAGE > 0:
        train_set = Subset(train_set, range(MAX_TRAIN_IMAGE))
    if MAX_VAL_IMAGE > 0:
        val_set = Subset(val_set, range(MAX_VAL_IMAGE))

    train_loader = DataLoader(train_set, batch_size=args['batch_size'], shuffle=True, num_workers=2,
                              pin_memory=True, collate_fn=DenseCapDataset.collate_fn)

    iter_counter = 0
    best_map = 0.

    # use tensorboard to track the loss
    if USE_TB:
        writer = SummaryWriter()

    scaler = GradScaler()
    rng = np.random.default_rng()

    for epoch in range(MAX_EPOCHS):

        for batch, ((img, targets, info), (car_images, car_classes, car_cam_poses)) in enumerate(zip(train_loader, car_data_loader)):

            img = [img_tensor.to(device) for img_tensor in img]
            targets = [{k: v.to(device) for k, v in target.items()} for target in targets]

            model.train()
            with autocast():
                losses = model(img, targets)

            detect_loss = losses['loss_objectness'] + losses['loss_rpn_box_reg'] + \
                          losses['loss_classifier'] + losses['loss_box_reg']
            caption_loss = losses['loss_caption']

            total_loss = args['detect_loss_weight'] * detect_loss + args['caption_loss_weight'] * caption_loss

            # auxillary loss
            # model.eval()
            # # TODO MARKUS: freeze roi heads
            # # TODO MARKUS: use autocast, scale losses
            # classes = np.array([CLASS_MAPPING[cls[0]] for cls in classes]).astype(int)
            #
            # features = []
            #
            # for img in car_images:
            #     # TODO MARKUS: what is the output?!
            #     predictions, img_features = model(img)
            #     features.append(img_features["p7"])
            #
            # features = torch.stack(features)
            #
            # contrastive_loss = torch.zeros(()).to(model.device)
            # multiview_loss = torch.zeros(()).to(model.device)
            #
            # for i in range(len(features)):
            #     i_features: torch.Tensor = f.normalize(torch.flatten(features[i]), dim=0)
            #     i_cam_pos = car_cam_poses[i]
            #
            #     i_class = classes[i]
            #     other_class_mask = classes != i_class
            #     same_class_mask = ~other_class_mask
            #
            #     if np.sum(same_class_mask) < 2:
            #         print("not enough same class samples, skipping..")
            #         continue
            #
            #     other_class_features = features[other_class_mask]
            #     same_class_features = features[same_class_mask]
            #
            #     # TODO: preferably sample from all three other classes..
            #     try:
            #         # this follows towers of babel
            #         # TODO do not simply take first same class feature
            #         same_feature_vec = f.normalize(torch.flatten(same_class_features[0]), dim=0)
            #         same_class_sim = torch.exp(torch.dot(i_features, same_feature_vec))
            #
            #         other_samples_idx = rng.choice(np.arange(len(other_class_features)), size=3, replace=False)
            #         other_samples = other_class_features[other_samples_idx]
            #
            #         other_class_sim_sum = 0
            #         for vec in other_samples:
            #             other_class_sim_sum += torch.exp(torch.dot(
            #                 i_features,
            #                 f.normalize(torch.flatten(vec), dim=0)
            #             ))
            #
            #         contrastive_loss += -torch.log(same_class_sim/(same_class_sim + other_class_sim_sum))
            #     except ValueError as e:
            #         print(e)
            #         continue
            #
            #     for j in range(i+1, len(features)):
            #         j_features: torch.Tensor = f.normalize(torch.flatten(features[j]), dim=0)
            #         j_cam_pos = car_cam_poses[j]
            #
            #         cam_distance = torch.linalg.vector_norm(i_cam_pos - j_cam_pos)
            #         dot = torch.dot(i_features, j_features)
            #
            #         # if cam_distance < 2:
            #         # dot product distance is most similar the greater it is..
            #         multiview_loss += torch.abs(dot / cam_distance)
            #
            # losses = contrastive_loss + multiview_loss

            # record loss
            if USE_TB:
                writer.add_scalar('batch_loss/total', total_loss.item(), iter_counter)
                writer.add_scalar('batch_loss/detect_loss', detect_loss.item(), iter_counter)
                writer.add_scalar('batch_loss/caption_loss', caption_loss.item(), iter_counter)

                writer.add_scalar('details/loss_objectness', losses['loss_objectness'].item(), iter_counter)
                writer.add_scalar('details/loss_rpn_box_reg', losses['loss_rpn_box_reg'].item(), iter_counter)
                writer.add_scalar('details/loss_classifier', losses['loss_classifier'].item(), iter_counter)
                writer.add_scalar('details/loss_box_reg', losses['loss_box_reg'].item(), iter_counter)

            if iter_counter % (len(train_set) / (args['batch_size'] * 16)) == 0:
                print("[{}][{}]\ntotal_loss {:.3f}".format(epoch, batch, total_loss.item()))
                for k, v in losses.items():
                    print(" <{}> {:.3f}".format(k, v))

            optimizer.zero_grad()
            # total_loss.backward()
            # apex backward
            scaler.scale(total_loss).backward()
            optimizer.step()

            if iter_counter > 0 and iter_counter % 2000 == 0:
                try:
                    results = quantity_check(model, val_set, idx_to_token, device, max_iter=100, verbose=True)
                    if results['map'] > best_map:
                        best_map = results['map']
                        save_model(model, optimizer, scaler, results, iter_counter)

                    if USE_TB:
                        writer.add_scalar('metric/map', results['map'], iter_counter)
                        writer.add_scalar('metric/det_map', results['detmap'], iter_counter)

                except AssertionError as e:
                    print('[INFO]: evaluation failed at epoch {}'.format(epoch))
                    print(e)

            iter_counter += 1

    save_model(model, optimizer, scaler, results, iter_counter, flag='end')

    if USE_TB:
        writer.close()


if __name__ == '__main__':
    args = set_args()
    train(args)
