import json
import os
from typing import List

import numpy as np
import torch
import torch.nn.functional as f
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision.models._api import get_weight
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn

from evaluate import quantity_check
from model.densecap import densecap_resnet50_fpn
from utils.data_loader import DenseCapDataset
from utils.three_dance_data import CarClassImageDataset

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

torch.backends.cudnn.benchmark = True
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_EPOCHS = 10
USE_TB = True
CONFIG_PATH = './model_params'
MODEL_NAME = 'without_aux'
IMG_DIR_ROOT = '../3-dance/datasets/vg'
VG_DATA_PATH = './data/VG-regions-lite.h5'
LOOK_UP_TABLES_PATH = './data/VG-regions-dicts-lite.pkl'
MAX_TRAIN_IMAGE = -1  # if -1, use all images in train set
MAX_VAL_IMAGE = -1

CLASSES = {0: 'BACK_LEFT', 1: 'BACK_RIGHT', 2: 'FRONT_LEFT', 3: 'FRONT_RIGHT'}
CLASS_MAPPING = {v: k for k, v in CLASSES.items()}


# TODO: add proper args parser
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
    args['multiview_loss_weight'] = 0.1
    args['contrastive_loss_weight'] = 1.0
    args['lr'] = 1e-4
    args['caption_lr'] = 1e-3
    args['weight_decay'] = 0
    args['batch_size'] = 4
    args['use_pretrain_fasterrcnn'] = True
    args['box_detections_per_img'] = 50
    args['train_auxiliary_loss'] = False

    if not os.path.exists(os.path.join(CONFIG_PATH, MODEL_NAME)):
        os.makedirs(os.path.join(CONFIG_PATH, MODEL_NAME))
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


def contrastive_clip_loss_fn(embeddings: torch.Tensor, learning_temp=np.log(1/0.07)):
    """
    Symmetric contrastive loss as introduced in CLIP.
    default learning_temp taken from clip implementation.
    """
    loss_fn = nn.CrossEntropyLoss()
    # class_sort = np.argsort(classes[indices])
    # embeddings = features[other_samples_idx]
    embeddings = f.normalize(torch.flatten(embeddings, start_dim=1), dim=1)

    logits = torch.matmul(embeddings, embeddings.T) * np.exp(learning_temp)

    # (not very) symmetric loss function, as cols and rows are equal!
    pseudo_labels = torch.arange(len(CLASSES.keys()), dtype=torch.long, device=device)
    loss = loss_fn(logits, pseudo_labels)

    return loss


def multiview_loss_fn(features: torch.Tensor, cam_poses):
    i_features: torch.Tensor = f.normalize(torch.flatten(features[0]), dim=0)
    i_cam_pos = cam_poses[0]
    loss = torch.zeros(()).to(device)

    for j in range(1, len(features)):
        j_features: torch.Tensor = f.normalize(torch.flatten(features[j]), dim=0)
        j_cam_pos = cam_poses[j]

        cam_distance = torch.linalg.vector_norm(i_cam_pos - j_cam_pos)
        dot = torch.dot(i_features, j_features)

        # if cam_distance < 2:
        # dot product distance is most similar the greater it is..
        loss += torch.exp(dot) / cam_distance

    return loss


def contrastive_multiview_loss(features: torch.Tensor, cam_poses, i: int, same_class_index: int,
                               other_samples_idx: List[int]):
    contrastive_loss = torch.zeros(()).to(device)
    multiview_loss = torch.zeros(()).to(device)

    i_features: torch.Tensor = f.normalize(torch.flatten(features[i]), dim=0)
    i_cam_pos = cam_poses[i]

    same_class_features = features[same_class_index]

    # this follows towers of babel
    same_feature_vec = f.normalize(torch.flatten(same_class_features), dim=0)
    same_class_sim = torch.exp(torch.dot(i_features, same_feature_vec))
    other_samples = features[other_samples_idx]

    other_class_sim_sum = 0
    for vec in other_samples:
        other_class_sim_sum += torch.exp(torch.dot(
            i_features,
            f.normalize(torch.flatten(vec), dim=0)
        ))

    contrastive_loss += -torch.log(same_class_sim / (same_class_sim + other_class_sim_sum))

    for j in range(len(features)):
        if i == j:
            continue
        j_features: torch.Tensor = f.normalize(torch.flatten(features[j]), dim=0)
        j_cam_pos = cam_poses[j]

        cam_distance = torch.linalg.vector_norm(i_cam_pos - j_cam_pos)
        dot = torch.dot(i_features, j_features)

        # if cam_distance < 2:
        # dot product distance is most similar the greater it is..
        multiview_loss += torch.abs(dot / cam_distance)

    return contrastive_loss, multiview_loss


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

    car_dataset = CarClassImageDataset(img_dir='../3-dance/data/car_images_model_sort')
    car_data_loader = DataLoader(car_dataset, batch_size=1, shuffle=True)

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
        for batch, data in enumerate(
                zip(train_loader, car_data_loader) if args['train_auxiliary_loss'] else train_loader):
            if args['train_auxiliary_loss']:
                (img, targets, info), (car_images, car_classes, car_cam_poses, model_name) = data
            else:
                img, targets, info = data

            img = [img_tensor.to(device) for img_tensor in img]
            targets = [{k: v.to(device) for k, v in target.items()} for target in targets]

            model.train()
            # with autocast():
            losses = model(img, targets)

            detect_loss = losses['loss_objectness'] + losses['loss_rpn_box_reg'] + \
                          losses['loss_classifier'] + losses['loss_box_reg']
            caption_loss = losses['loss_caption']

            total_loss = args['detect_loss_weight'] * detect_loss + args['caption_loss_weight'] * caption_loss

            # auxiliary loss
            if args['train_auxiliary_loss']:
                # TODO MARKUS: freeze roi heads
                # TODO MARKUS: use autocast, scale losses
                car_classes = np.array([CLASS_MAPPING[cls[0]] for cls in car_classes]).astype(int)

                # skip invalid i's
                while True:
                    i = rng.choice(np.arange(len(car_images)), 1).item()
                    # for i in range(len(features)):
                    i_class = car_classes[i]
                    other_samples_idx = []
                    try:
                        for cls in CLASSES.keys():
                            if cls == i_class:
                                continue

                            class_mask = car_classes == cls
                            class_indices = class_mask.nonzero()[0]
                            samples_index = rng.choice(class_indices, size=1).item()
                            other_samples_idx.append(samples_index)
                    except ValueError as e:
                        print(e)
                        continue

                    other_class_mask = car_classes != i_class
                    # same_class_indices = (~other_class_mask).nonzero()[0]
                    # # make sure we do not compare to ith features
                    # same_class_indices = same_class_indices[same_class_indices != i]
                    #
                    # if len(same_class_indices) < 1:
                    #     print("not enough same class samples, skipping..")
                    #     continue

                    break

                if False:
                    features = []
                    # disable grad for other calls than i'th!
                    for idx, img in enumerate(car_images):
                        if idx != i:
                            with torch.no_grad():
                                # with autocast():
                                backbone_features = model.backbone(img.to(device))
                                features.append(backbone_features['pool'])
                        else:
                            backbone_features = model.backbone(img.to(device))
                            features.append(backbone_features['pool'])

                    features = torch.stack(features)

                    same_class_index = rng.choice(same_class_indices, 1).item()
                    contrastive_loss, multiview_loss = contrastive_multiview_loss(features, car_cam_poses, i,
                                                                              same_class_index, other_samples_idx)

                    auxiliary_losses = args['contrastive_loss_weight'] * contrastive_loss + \
                                       args['multiview_loss_weight'] * multiview_loss

                else:
                    other_samples_idx.insert(0, i)
                    cam_poses = []
                    for idx in other_samples_idx:
                        cam_poses.append(car_cam_poses[idx].to(device))

                    output = model.backbone(car_images[0][other_samples_idx].to(device))
                    embeddings = output['pool']

                    contrastive_loss = contrastive_clip_loss_fn(embeddings)
                    multiview_loss = multiview_loss_fn(embeddings, cam_poses)
                    auxiliary_losses = args['contrastive_loss_weight'] * contrastive_loss + args['multiview_loss_weight'] * multiview_loss

            # record loss
            if USE_TB:
                writer.add_scalar('batch_loss/total', total_loss.item(), iter_counter)
                writer.add_scalar('batch_loss/detect_loss', detect_loss.item(), iter_counter)
                writer.add_scalar('batch_loss/caption_loss', caption_loss.item(), iter_counter)

                writer.add_scalar('details/loss_objectness', losses['loss_objectness'].item(), iter_counter)
                writer.add_scalar('details/loss_rpn_box_reg', losses['loss_rpn_box_reg'].item(), iter_counter)
                writer.add_scalar('details/loss_classifier', losses['loss_classifier'].item(), iter_counter)
                writer.add_scalar('details/loss_box_reg', losses['loss_box_reg'].item(), iter_counter)

                if args['train_auxiliary_loss']:
                    writer.add_scalar('batch_loss/auxiliary_losses', auxiliary_losses.item(), iter_counter)
                    writer.add_scalar('details/contrastive_loss', contrastive_loss.item(), iter_counter)
                    writer.add_scalar('details/multiview_loss', multiview_loss.item(), iter_counter)

            if iter_counter % (len(train_set) / (args['batch_size'] * 16)) == 0:
                print("[{}][{}]\ntotal_loss {:.3f}".format(epoch, batch, total_loss.item()))
                for k, v in losses.items():
                    print(" <{}> {:.3f}".format(k, v))

            optimizer.zero_grad()
            # total_loss.backward()
            # apex backward
            if args['train_auxiliary_loss']:
                total_loss.backward()
                auxiliary_losses.backward()
                # scaler.scale(total_loss + auxiliary_losses).backward()
            else:
                total_loss.backward()
                # scaler.scale(total_loss).backward()
            optimizer.step()
            # scaler.step(optimizer)
            # scaler.update()

            if iter_counter > 0 and iter_counter % 4000 == 0:
                try:
                    with autocast():
                        results = quantity_check(model, val_set, idx_to_token, device, verbose=False)
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
