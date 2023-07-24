from typing import List
from torch import nn
import torch
import torch.nn.functional as F
from torchvision.models._api import WeightsEnum, get_weight

from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from model.box_describer import BoxDescriber
from model.roi_heads import DenseCapRoIHeads
from preprocess import encode_caption, words_preprocess


__all__ = [
    "DenseCapModel", "densecap_resnet50_fpn",
]


class DenseCapModel(GeneralizedRCNN):
    def __init__(self, backbone, return_features = False,
                 # Caption parameters
                 box_describer=None,
                 feat_size=None, hidden_size=None, max_len=None,
                 emb_size=None, rnn_num_layers=None, vocab_size=None,
                 fusion_type='init_inject',
                 # transform parameters
                 min_size=300, max_size=720,  # 300不确定
                 image_mean=None, image_std=None,
                 # RPN parameters
                 rpn_anchor_generator=None, rpn_head=None,
                 rpn_pre_nms_top_n_train=250, rpn_pre_nms_top_n_test=100,
                 rpn_post_nms_top_n_train=250, rpn_post_nms_top_n_test=100,
                 rpn_nms_thresh=0.7,
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
                 # Box parameters
                 view_head=None, n_views=8,                
                 box_roi_pool=None, box_head=None, box_predictor=None,
                 box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                 box_batch_size_per_image=512, box_positive_fraction=0.25,
                 bbox_reg_weights=None,
                 fixed_size=(512, 512),
                 token_to_idx=None,
                 losses = [],
                 name="densecap"):

        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels "
                "specifying the number of output channels (assumed to be the "
                "same for all the levels)")

        assert isinstance(rpn_anchor_generator, (AnchorGenerator, type(None)))
        assert isinstance(box_roi_pool, (MultiScaleRoIAlign, type(None)))

        if box_describer is None:
            for param in {max_len, emb_size, rnn_num_layers, vocab_size}:
                assert isinstance(param, int) and param > 0, 'invalid parameters of caption'
        else:
            assert max_len is None and emb_size is None and rnn_num_layers is None and vocab_size is None

        out_channels = backbone.out_channels

        if rpn_anchor_generator is None:
            anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            rpn_anchor_generator = AnchorGenerator(
                anchor_sizes, aspect_ratios
            )
        if rpn_head is None:
            rpn_head = RPNHead(
                out_channels, rpn_anchor_generator.num_anchors_per_location()[0]
            )

        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        rpn = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh)

        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(
                featmap_names=['0', '1', '2', '3'],
                output_size=7,
                sampling_ratio=2)

        if box_head is None:
            resolution = box_roi_pool.output_size[0]
            representation_size = 4096 if feat_size is None else feat_size
            box_head = TwoMLPHead(
                out_channels * resolution ** 2,
                representation_size)
            
        if view_head is None:
            representation_size = 4096 if feat_size is None else feat_size
            view_head = ViewHead(
                representation_size,
                64,
                n_views)

        if box_predictor is None:
            representation_size = 4096 if feat_size is None else feat_size
            box_predictor = FastRCNNPredictor(
                representation_size,
                2)

        if box_describer is None:
            representation_size = 4096 if feat_size is None else feat_size
            box_describer = BoxDescriber(representation_size, hidden_size, max_len,
                                         emb_size, rnn_num_layers, vocab_size, fusion_type)

        roi_heads = DenseCapRoIHeads(
            # Caption
            box_describer,
            # Box
            view_head,
            box_roi_pool, box_head, box_predictor,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_batch_size_per_image, box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh, box_nms_thresh, box_detections_per_img,
            # Whether return features during testing
            losses,
            return_features)

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std, fixed_size=fixed_size)

        self.token_to_idx = token_to_idx
        self.name = name

        super(DenseCapModel, self).__init__(backbone, rpn, roi_heads, transform)


    def toDevice(self, device):
        self.device = device
        self.roi_heads.device = device
        self.to(device)
        
    # TODO: do this prior to training time
    def tokenize(self, captions: List[str]):
        tokenized_captions = []

        for caption in captions:
            caption = words_preprocess(caption)
            tokenized = encode_caption(caption, self.token_to_idx, max_token_length=19)
            tokenized_captions.append(torch.tensor(tokenized, dtype=torch.long, device=self.device))

        return torch.stack(tokenized_captions)


    def query_caption(self, target_images: List[torch.Tensor], captions: List[str], views: List[int]):        
        images, _ = self.transform(target_images, None)
        tokenized_captions = self.tokenize(captions)

        features = self.backbone(images.tensors)
        proposals, _ = self.rpn(images, features, None)
        losses = self.roi_heads.forward_query(features, proposals, images.image_sizes, tokenized_captions, views)

        return losses



class ViewHead(nn.Module):
    """
    Standard heads for FPN-based models

    Arguments:
        in_channels (int): number of input channels
        hidden_size (int): hidden layer size
        n_classes (int): numer of output classes
    """

    def __init__(self, in_channels, hidden_size, n_classes):
        super(ViewHead, self).__init__()

        self.fc6 = nn.Linear(in_channels, hidden_size)
        self.fc7 = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        x = x.flatten(start_dim=1)

        x = F.relu(self.fc6(x))
        x = self.fc7(x)

        return x


class TwoMLPHead(nn.Module):
    """
    Standard heads for FPN-based models

    Arguments:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """

    def __init__(self, in_channels, representation_size):
        super(TwoMLPHead, self).__init__()

        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x


class FastRCNNPredictor(nn.Module):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.

    Arguments:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels, num_classes):
        super(FastRCNNPredictor, self).__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas


def densecap_resnet50_fpn(backbone_pretrained=False, **kwargs):
    weights = get_weight("ResNet50_Weights.DEFAULT") if backbone_pretrained else None
    backbone = resnet_fpn_backbone(backbone_name='resnet50', weights=weights, trainable_layers=5)
    model = DenseCapModel(backbone, **kwargs)

    return model
