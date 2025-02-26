import json
import os
from pathlib import Path

import torch

from model.densecap import DenseCapModel, densecap_resnet50_fpn


def load_model(
    model_config_path: Path,
    checkpoint_path: Path,
    return_features=False,
    box_per_img=50,
    verbose=False,
    **kwargs,
):
    with open(model_config_path, "r") as f:
        model_args = json.load(f)

    model = densecap_resnet50_fpn(
        backbone_pretrained=model_args["backbone_pretrained"],
        return_features=return_features,
        feat_size=model_args["feat_size"],
        hidden_size=model_args["hidden_size"],
        max_len=model_args["max_len"],
        emb_size=model_args["emb_size"],
        rnn_num_layers=model_args["rnn_num_layers"],
        vocab_size=model_args["vocab_size"],
        fusion_type=model_args["fusion_type"],
        box_detections_per_img=box_per_img,
        **kwargs,
    )

    checkpoint = torch.load(checkpoint_path)
    model_dict = checkpoint["model"]
    # adjust to previous model layer names
    for k in list(model_dict.keys()):
        if "roi_heads.view_head" in k:
            model_dict[
                k.replace("roi_heads.view_head", "roi_heads.region_view_head")
            ] = model_dict.pop(k)
        if "roi_heads.view_predictor_head" in k:
            model_dict[
                k.replace(
                    "roi_heads.view_predictor_head", "roi_heads.caption_view_predictor"
                )
            ] = model_dict.pop(k)

    model.load_state_dict(checkpoint["model"], strict=False)

    if verbose and "results_on_val" in checkpoint.keys():
        print("[INFO]: checkpoint {} loaded".format(checkpoint_path))
        print("[INFO]: correspond performance on val set:")
        for k, v in checkpoint["results_on_val"].items():
            if not isinstance(v, dict):
                print("        {}: {:.3f}".format(k, v))

    return model


def save_model(
    model: DenseCapModel,
    optimizer,
    scaler,
    results_on_val,
    iter_counter,
    flag=None,
    prefix=None,
):
    if not os.path.exists("model_params"):
        os.makedirs("model_params")

    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "results_on_val": results_on_val,
        "iterations": iter_counter,
    }

    filename = os.path.join(
        "model_params",
        f"{f'{prefix}_' if isinstance(prefix, str) else ''}{model.name}{f'_{flag}' if isinstance(flag, str) else ''}.pth.tar",
    )

    print(f"Saving checkpoint to {filename}")
    torch.save(state, filename)
