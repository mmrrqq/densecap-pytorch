import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from torchvision.ops import boxes as box_ops
from torchvision.models.detection import _utils as det_utils


def predict_view_loss(view_predicts, gt_views):        
    view_loss = F.cross_entropy(view_predicts, gt_views)

    return view_loss


def detect_loss(class_logits, box_regression, labels, regression_targets):
    """
    Computes the loss for detection part.
    Arguments:
        class_logits (Tensor)
        box_regression (Tensor)
        labels (list[BoxList])
        regression_targets (Tensor)
    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
    """

    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    classification_loss = F.cross_entropy(class_logits, labels)

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
    labels_pos = labels[sampled_pos_inds_subset]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, -1, 4)

    box_loss = F.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        reduction="sum",
    )
    box_loss = box_loss / labels.numel()

    return classification_loss, box_loss


def query_caption_loss(caption_predicts, caption_gt) -> torch.Tensor:
    """
    Computes the loss for caption part.
    Arguments:
        caption_predicts (Tensor)
        caption_gt (Tensor or list[Tensor])
        caption_length (Tensor or list[Tensor])
        caption_loss (Tensor)
    """

    caption_gt = caption_gt[:, 1:]    

    return F.cross_entropy(caption_predicts.permute((0, 2, 1)), caption_gt, reduction='none')


def caption_loss(caption_predicts, caption_gt, caption_length):
    """
    Computes the loss for caption part.
    Arguments:
        caption_predicts (Tensor)
        caption_gt (Tensor or list[Tensor])
        caption_length (Tensor or list[Tensor])
        caption_loss (Tensor)
    """

    if isinstance(caption_gt, list) and isinstance(caption_length, list):
        caption_gt = torch.cat(caption_gt, dim=0)  # (batch_size, max_len+1)
        caption_length = torch.cat(caption_length, dim=0) # (batch_size, )
        assert caption_predicts.shape[0] == caption_gt.shape[0] and caption_predicts.shape[0] == caption_length.shape[0]

    # '<bos>' is not considered
    caption_length = torch.clamp(caption_length-1, min=0).cpu()

    predict_pps = pack_padded_sequence(caption_predicts, caption_length, batch_first=True, enforce_sorted=False)

    target_pps = pack_padded_sequence(caption_gt[:, 1:], caption_length, batch_first=True, enforce_sorted=False)    

    return F.cross_entropy(predict_pps.data, target_pps.data)


class DenseCapRoIHeads(nn.Module):

    def __init__(self,
                 box_describer,
                 view_head,
                 box_roi_pool,
                 box_head,
                 box_predictor,
                 # Faster R-CNN training
                 fg_iou_thresh, bg_iou_thresh,
                 batch_size_per_image, positive_fraction,
                 bbox_reg_weights,
                 # Faster R-CNN inference
                 score_thresh,
                 nms_thresh,
                 detections_per_img,
                 # Whether return features during testing
                 return_features=False,
                 ):

        super(DenseCapRoIHeads, self).__init__()

        self.return_features = return_features
        self.box_similarity = box_ops.box_iou
        # assign ground-truth boxes for each proposal
        self.proposal_matcher = det_utils.Matcher(
            fg_iou_thresh,
            bg_iou_thresh,
            allow_low_quality_matches=False)

        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(
            batch_size_per_image,
            positive_fraction)

        if bbox_reg_weights is None:
            bbox_reg_weights = (10., 10., 5., 5.)
        self.box_coder = det_utils.BoxCoder(bbox_reg_weights)

        self.box_roi_pool = box_roi_pool
        self.box_head = box_head
        self.box_predictor = box_predictor
        self.box_describer = box_describer

        self.view_head = view_head

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img

    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels):

        matched_idxs = []
        labels = []
        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image in zip(proposals, gt_boxes, gt_labels):  # 每张图片循环

            device = proposals_in_image.device
            if gt_boxes_in_image.numel() == 0:
                # Background image
                clamped_matched_idxs_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
                labels_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
            else:
                #  set to self.box_similarity when https://github.com/pytorch/pytorch/issues/27495 lands
                match_quality_matrix = box_ops.box_iou(gt_boxes_in_image, proposals_in_image)
                # iou (Tensor[N, M]): the NxM matrix containing the IoU values for every element in boxes1 and boxes2

                matched_idxs_in_image = self.proposal_matcher(match_quality_matrix)

                clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)

                labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]

                # Label background (below the low threshold)
                bg_inds = matched_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD
                labels_in_image[bg_inds] = torch.tensor(0, device=device)

                # Label ignore proposals (between low and high thresholds)
                ignore_inds = matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLDS
                labels_in_image[ignore_inds] = torch.tensor(-1, device=device)  # -1 is ignored by sampler

            matched_idxs.append(clamped_matched_idxs_in_image)
            labels.append(labels_in_image)
        return matched_idxs, labels

    def subsample(self, labels):

        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_inds = []
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(
                zip(sampled_pos_inds, sampled_neg_inds)
        ):
            img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
            sampled_inds.append(img_sampled_inds)
        return sampled_inds

    def select_training_samples(self, proposals, targets):
        """
        proposals: (List[Tensor[N, 4]])
        targets (List[Dict])
        """
        assert targets is not None
        dtype = proposals[0].dtype
        device = proposals[0].device

        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_captions = [t["caps"] for t in targets]
        gt_captions_length = [t["caps_len"].to(device) for t in targets]
        gt_labels = [torch.ones((t["boxes"].shape[0],), dtype=torch.int64, device=device) for t in
                     targets]  # generate labels LongTensor(1)
        gt_views = [t["view"] if "view" in t else None for t in targets]

        # append ground-truth bboxes to propos
        # List[2*N,4],一个list是一张图片
        proposals = [
            torch.cat((proposal, gt_box))
            for proposal, gt_box in zip(proposals, gt_boxes)
        ]

        # get matching gt indices for each proposal
        matched_idxs, labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)
        # sample a fixed proportion of positive-negative proposals
        sampled_inds = self.subsample(labels)
        matched_gt_boxes = []
        num_images = len(proposals)
        for img_id in range(num_images):
            img_sampled_inds = sampled_inds[img_id]  # (M,) 0~P-1
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            labels[img_id] = labels[img_id][img_sampled_inds]
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]  # before (P,) / after (M,) 0~N-1

            gt_boxes_in_image = gt_boxes[img_id]
            if gt_boxes_in_image.numel() == 0:
                gt_boxes_in_image = torch.zeros((1, 4), dtype=dtype, device=device)
            matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])
            gt_captions[img_id] = gt_captions[img_id][matched_idxs[img_id]]  # before (N, ) / after (M, )
            gt_captions_length[img_id] = gt_captions_length[img_id][matched_idxs[img_id]]

        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)

        return proposals, matched_idxs, gt_captions, gt_captions_length, labels, regression_targets, gt_views

    def postprocess_detections(self, logits, box_regression, caption_predicts, proposals, image_shapes,
                               box_features, view_predicts, return_features):
        device = logits.device
        num_classes = logits.shape[-1]

        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)
        
        pred_scores = F.softmax(logits, -1)        
        view_predicts = F.softmax(view_predicts, -1)        

        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)
        pred_caption_list = caption_predicts.split(boxes_per_image, 0)        
        pred_predicts_list = view_predicts.split(boxes_per_image, 0)
        if return_features:
            pred_box_features_list = box_features.split(boxes_per_image, 0)
        else:
            pred_box_features_list = None

        all_boxes = []
        all_scores = []
        all_labels = []
        all_captions = []
        all_box_features = []
        all_view_predicts = []
        remove_inds_list = []
        keep_list = []
        for boxes, scores, captions, image_shape, views in zip(pred_boxes_list, pred_scores_list, pred_caption_list,
                                                        image_shapes, pred_predicts_list):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)            

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)            

            # remove low scoring boxes
            inds = torch.nonzero(scores > self.score_thresh).squeeze(1)
            remove_inds_list.append(inds)
            boxes, scores, captions, labels, views = boxes[inds], scores[inds], captions[inds], labels[inds], views[inds]

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, captions, labels, views = boxes[keep], scores[keep], captions[keep], labels[keep], views[keep]

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[:self.detections_per_img]
            keep_list.append(keep)
            boxes, scores, captions, labels, views = boxes[keep], scores[keep], captions[keep], labels[keep], views[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_captions.append(captions)
            all_labels.append(labels)
            all_view_predicts.append(views)

        if return_features:
            for inds, keep, box_features in zip(remove_inds_list, keep_list, pred_box_features_list):
                all_box_features.append(box_features[inds[keep]//(num_classes-1)])

        return all_boxes, all_scores, all_captions, all_box_features, all_view_predicts

    def forward(self, features, proposals, image_shapes, targets=None):
        """
        Arguments:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        if targets is not None:
            for t in targets:
                floating_point_types = (torch.float, torch.double, torch.half)
                t["caps_len"] = t["caps_len"].cpu()
                assert t["boxes"].dtype in floating_point_types, 'target boxes must of float type'
                assert t["caps"].dtype == torch.int64, 'target caps must of int64 (torch.long) type'
                assert t["caps_len"].dtype == torch.int64, 'target caps_len must of int64 (torch.long) type'
                assert t["caps_len"].device == torch.device("cpu"), 'target caps_len must be cpu tensor'

        if self.training:
            proposals, matched_idxs, caption_gt, caption_length, labels, regression_targets, gt_views = \
                self.select_training_samples(proposals, targets)                        
        else:
            labels = None
            matched_idxs = None
            caption_gt = None
            caption_length = None
            regression_targets = None
            gt_views = None

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        logits, box_regression = self.box_predictor(box_features)                

        if self.training:
            # labels 到这里应该是有0和（1，class-1），0代表背景，其余代表类别，需要剔除背景，然后进行描述(List[Tensor])
            # 也需要滤除对应的caption和caption_length
            keep_ids = [label>0 for label in labels]
            boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]            
            box_features = box_features.split(boxes_per_image, 0)
            gt_views = [v.repeat(n_boxes) for n_boxes, v in zip(boxes_per_image, gt_views)]        
            box_features_gt = []
            for i in range(len(keep_ids)):
                box_features_gt.append(box_features[i][keep_ids[i]])
                caption_gt[i] = caption_gt[i][keep_ids[i]]
                caption_length[i] = caption_length[i][keep_ids[i]]
                gt_views[i] = gt_views[i][keep_ids[i]]
            box_features = torch.cat(box_features_gt, 0)
            gt_views = torch.cat(gt_views, 0)
        
        view_predicts = self.view_head(box_features)
        caption_predicts = self.box_describer(box_features, caption_gt, caption_length)        

        result, losses = [], {}
        if self.training:
            loss_view_predictor = predict_view_loss(view_predicts, gt_views)
            loss_classifier, loss_box_reg = detect_loss(logits, box_regression, labels, regression_targets)
            loss_caption = caption_loss(caption_predicts, caption_gt, caption_length)

            losses = {
                "loss_view_predictor": loss_view_predictor,
                "loss_classifier": loss_classifier,
                "loss_box_reg": loss_box_reg,
                "loss_caption": loss_caption
            }
        else:
            boxes, scores, caption_predicts, feats, view_scores = self.postprocess_detections(logits, box_regression,
                                                                                 caption_predicts, proposals,
                                                                                 image_shapes, box_features, view_predicts,
                                                                                 self.return_features)
                        
            num_images = len(boxes)
            for i in range(num_images):                
                result.append(
                    {
                        "boxes": boxes[i],
                        "caps": caption_predicts[i],
                        "scores": scores[i],
                        "views": view_scores[i].argmax(dim=1)
                    }
                )
                if self.return_features:
                    result[-1]['feats'] = feats[i]

        return result, losses
    

    def forward_query(self, features, proposals, image_shapes, target_query, target_view):
        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)        
        # logits, box_regression = self.box_predictor(box_features)
        view_predicts = self.view_head(box_features)        

        batch_size, _ = box_features.shape                
        target_embeddings = target_query.expand(batch_size, -1)        
        target_lens = (target_query != 0).sum(dim=1)        
        target_lens = target_lens.expand(batch_size)
        
        caption_predicts = self.box_describer.forward_train(box_features, target_embeddings, target_lens)
        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]            

        view_predicts = view_predicts.split(boxes_per_image, 0)
        view_predicts = torch.cat(view_predicts, 0)
        gt_views = [v.repeat(n_boxes) for n_boxes, v in zip(boxes_per_image, target_view)]        
        gt_views = torch.cat(gt_views, 0).to(view_predicts.device)

        loss_view_predictor = predict_view_loss(view_predicts, gt_views)                        
        loss_caption = query_caption_loss(caption_predicts, target_embeddings)        
        
        box_mean_caption_loss = loss_caption.mean(dim=1)        

        index_mapping = torch.arange(len(box_mean_caption_loss))
        index_mapping = index_mapping.split(boxes_per_image, 0)
        index_mapping = torch.stack(index_mapping)        

        loss_caption_per_view = box_mean_caption_loss.split(boxes_per_image, 0)        
        loss_caption_per_view = torch.stack(loss_caption_per_view)

        min_loss_index_per_view = loss_caption_per_view.argmin(dim=1)
        min_loss_index_per_view = torch.tensor([index_mapping[i, idx] for i, idx in enumerate(min_loss_index_per_view)])        

        min_loss_index = box_mean_caption_loss.argmin()        

        min_caption_predicts = self.box_describer.forward_test(box_features[min_loss_index].unsqueeze(dim=0))
        min_caption_predicts_per_view = self.box_describer.forward_test(box_features[min_loss_index_per_view])
        losses = {
            "view": loss_view_predictor, 
            "caption_min": box_mean_caption_loss[min_loss_index],
            "caption_view_min": box_mean_caption_loss[min_loss_index_per_view]
            # "caption_mean": box_mean_caption_loss.mean(),
            # "caption_std": box_mean_caption_loss.std()
        }                

        return losses, (min_caption_predicts[0], min_caption_predicts_per_view)
           

