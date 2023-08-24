import argparse
from pathlib import Path
import pickle
import time
from typing import Optional

import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
from utils.model_io import load_model, save_model
from model.densecap import DenseCapModel
from utils.snare_dataset import SnareDataset

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# device = torch.device("cpu")
VIEW_HEAD_LR = 1e-3
CAP_LR = 1e-5
LR = 1e-4
WEIGHT_DECAY = 0
ACCUMULATE_BATCH_SIZE = 64


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-prefix", default=time.time())
    parser.add_argument("--params-path", default="model_params")
    parser.add_argument("--model-name", default="xx")
    parser.add_argument("--test-view", action="store_true", default=False)
    parser.add_argument("--test-iterations", default=10)
    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--alternating", action="store_true", default=False)
    parser.add_argument(
        "--losses",
        nargs="+",
        default=["view", "multiview", "view_contrastive", "min_cap", "multiview_cap"],
    )

    return parser.parse_args()


def print_decode_caption(cap: torch.Tensor, idx_to_token):
    for i in cap:
        if i < 1:
            break
        print(idx_to_token[i.item()], end=" ")

    print("\n")


def train(
    model: DenseCapModel,
    data_loader: DataLoader,
    iter_offset: int = 0,
    writer: Optional[SummaryWriter] = None,
    args = {}
):
    model.train()

    # freeze region proposals
    model.rpn.training = False
    for param in model.rpn.parameters():
        param.requires_grad = False

    for param in model.roi_heads.box_predictor.parameters():
        param.requires_grad = False

    # TEMP: freeze all layers except view predictors..
    # for name, param in model.named_parameters():
    #     if "box_describer" not in name and "view_head" not in name and "view_predictor_head" not in name:
    #         param.requires_grad = False

    # for param in model.roi_heads.box_describer.parameters():
    #     param.requires_grad = False
    # END TEMP: freeze all layers except view predictors..

    view_ids = torch.arange(8)    

    optimizer = torch.optim.AdamW(
        [
            {
                "params": (
                    para
                    for name, para in model.named_parameters()
                    if para.requires_grad
                    and "box_describer" not in name
                    and "view_head" not in name
                    and "view_predictor_head" not in name
                ),
                "name": "base",
            },
            {
                "params": (
                    para
                    for para in model.roi_heads.box_describer.parameters()
                    if para.requires_grad
                ),
                "lr": CAP_LR,
                "name": "captioning",
            },
            {
                "params": (
                    para
                    for para in model.roi_heads.view_head.parameters()
                    if para.requires_grad
                ),
                "lr": VIEW_HEAD_LR,
                "name": "view_head",
            },
            {
                "params": (
                    para
                    for para in model.roi_heads.view_predictor_head.parameters()
                    if para.requires_grad
                ) if not args.alternating else (),
                "lr": VIEW_HEAD_LR,
                "name": "view_prediction_head",
            }
        ],
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )    

    view_pred_optimizer = torch.optim.AdamW(
        [
            {
                "params": (
                    para
                    for name, para in model.named_parameters()
                    if para.requires_grad and "view_predictor_head" in name
                )
            }
        ],
        lr=VIEW_HEAD_LR        
    ) if args.alternating else None

    iter_count = iter_offset

    cap_view_pred_mode = False

    accum_dict = {}

    for i, batch in enumerate(tqdm(data_loader)):
        imgs, gt_idxs, _, annotation, is_visual = batch

        if is_visual.sum() < data_loader.batch_size or (gt_idxs < 0).sum() > 0:
            continue

        imgs = [img.squeeze().to(device) for img in imgs]
        # key2_imgs = [k.squeeze().to(device) for k in key2_imgs]

        loss, loss_dict, _ = model.query_caption(imgs, annotation, view_ids)        

        if cap_view_pred_mode:
            loss = loss_dict['view_prediction'] / ACCUMULATE_BATCH_SIZE            
        else:
            loss = loss / ACCUMULATE_BATCH_SIZE            

        loss.backward()

        for k, v in loss_dict.items():
            if accum_dict.get(k):
                accum_dict[k] += v.item() / ACCUMULATE_BATCH_SIZE
            else:
                accum_dict[k] = v.item() / ACCUMULATE_BATCH_SIZE

        if ((iter_count + 1) % ACCUMULATE_BATCH_SIZE == 0) or (
            iter_count + 1 == len(data_loader)
        ):
            if args.alternating:
                (view_pred_optimizer if cap_view_pred_mode else optimizer).step()
                (view_pred_optimizer if cap_view_pred_mode else optimizer).zero_grad()

                if not cap_view_pred_mode and i > (len(data_loader) / 2):
                    print("switch to cap view pred")
                    cap_view_pred_mode = True

            else:
                optimizer.step()
                optimizer.zero_grad()

            if writer is not None:
                # writer.add_scalar("batch_loss/total", loss.item(), iter_count)
                for k, v in accum_dict.items():            
                    writer.add_scalar(f"batch_loss/{k}", v, iter_count)

                accum_dict.clear()

        iter_count += 1

    return iter_count


def test(model: DenseCapModel, data_loader: DataLoader):
    model.eval()
    view_ids = torch.arange(8)

    n = 0
    min_pos = 0    
    mean_pos = 0    
    min_per_view_mean_pos = 0
    view_pred_correct = 0
    view_pred_total = 0
    cap_view_pred_correct = 0

    with torch.no_grad():
        for batch in tqdm(data_loader):
            (key1_imgs, key2_imgs), gt_idx, (key1, key2), annotation, is_visual = batch

            if not is_visual or gt_idx < 0:
                continue

            annotation = annotation[0]

            if gt_idx > 0:
                key1_imgs, key2_imgs = key2_imgs, key1_imgs

            key1_imgs = [k.squeeze().to(device) for k in key1_imgs]
            _, losses1, _ = model.query_caption(key1_imgs, [annotation], view_ids)

            del key1_imgs

            key2_imgs = [k.squeeze().to(device) for k in key2_imgs]
            _, losses2, _ = model.query_caption(key2_imgs, [annotation], view_ids)
            
            n += 1
            if losses2["cap_min"] > losses1["cap_min"]:
                min_pos += 1

            if losses2["cap_mean"] > losses1["cap_mean"]:
                mean_pos += 1            

            if losses2["cap_min_per_view_mean"] > losses1["cap_min_per_view_mean"]:
                min_per_view_mean_pos += 1            

            view_preds = losses1["view_preds"].cpu()
            view_pred_total += len(view_preds)
            view_pred_correct += (view_preds == view_ids).sum()
            cap_view_pred_correct += (losses1["cap_min_view"] == losses1["view_cap_preds"]).sum()

    mean_acc = mean_pos / n    
    min_acc = min_pos / n
    min_per_view_mean_acc = min_per_view_mean_pos / n    
    view_pred_acc = view_pred_correct / view_pred_total
    cap_view_pred_acc = cap_view_pred_correct / n

    print(
        f"test end.\nmin:\t{min_acc:.2f}\nmean:\t{mean_acc:.2f}\nmin per view mean:\t{min_per_view_mean_acc}\nview pred:\t{view_pred_acc}\ncap view pred:\t{cap_view_pred_acc}"
    )
    return {
        "min_acc": min_acc,
        "mean_acc": mean_acc,
        "min_per_view_mean_acc": min_per_view_mean_acc,
        "view_pred_acc": view_pred_acc,
        "cap_view_pred_acc": cap_view_pred_acc
    }


def view_predict(model: DenseCapModel, images, query: str, best_view_id: int):
    """Select a random image, predict view and the best view according to the query and 
    calculate and return query probability on the predicted best view.
    """
    images = [k.squeeze().to(device) for k in images]    
    rnd_view_id: int = torch.randint(low=0, high=len(images), size=(1,)).item()
    rnd_img = images[rnd_view_id]    

    pred_best_img = images[best_view_id]
    _, best_view_losses, _ = model.query_caption([pred_best_img], [query], torch.tensor([best_view_id]))
    _, random_view_losses, _ = model.query_caption([rnd_img], [query], torch.tensor([rnd_view_id]))

    return best_view_losses, random_view_losses


def test_view_prediction(model: DenseCapModel, data_loader: DataLoader, iterations=10):
    model.eval()        

    rnd_accs = []
    pred_accs = []

    with torch.no_grad():
        for _ in range(iterations):
            n = 0
            pred_pos = 0    
            random_pos = 0
            torch.random.seed()

            for i, batch in enumerate(tqdm(data_loader)):
                (key1_imgs, key2_imgs), gt_idx, (key1, key2), annotation, is_visual = batch

                if not is_visual or gt_idx < 0:
                    continue

                annotation = annotation[0]

                if gt_idx > 0:
                    key1_imgs, key2_imgs = key2_imgs, key1_imgs

                key1_imgs = [k.squeeze().to(device) for k in key1_imgs]
                best_view_id = model.query_view_caption(annotation).cpu().item()                
                best_view_losses1, random_losses1 = view_predict(model, key1_imgs, annotation, best_view_id)

                del key1_imgs

                key2_imgs = [k.squeeze().to(device) for k in key2_imgs]
                best_view_losses2, random_losses2 = view_predict(model, key2_imgs, annotation, best_view_id)
                
                n += 1
                if best_view_losses2["cap_min"] > best_view_losses1["cap_min"]:
                    pred_pos += 1

                if random_losses2["cap_min"] > random_losses1["cap_min"]:
                    random_pos += 1

                if i % 100 == 0:
                    pred_acc = pred_pos / n
                    random_acc = random_pos / n    

                    print(
                        f"intermedia results\npred:\t{pred_acc:.2f}\nrandom:\t{random_acc:.2f}"
                    )
        
            pred_acc = pred_pos / n
            random_acc = random_pos / n

            pred_accs.append(pred_acc)
            rnd_accs.append(random_acc)

            print(
                f"view prediction test end.\npred:\t{pred_acc:.2f}\nrandom:\t{random_acc:.2f}"
            )

    print(pred_accs)
    print(rnd_accs)


def train_loop(args):
    print(args.losses)
    lut_path = Path("./data/VG-regions-dicts-lite.pkl")

    with open(lut_path, "rb") as f:
        look_up_tables = pickle.load(f)
    
    token_to_idx = look_up_tables["token_to_idx"]

    params_path = Path("compute_model_params")
    model_name = "without_aux"
    model = load_model(
        params_path / model_name / "config.json",
        params_path / (model_name + ".pth.tar"),
        return_features=False,
        losses=args.losses,
    )
    model.name = "_".join(args.losses)
    model.token_to_idx = token_to_idx

    model.toDevice(device)
    test_set = SnareDataset(mode="valid", filter_visual=True)

    train_set = SnareDataset(mode="train", filter_visual=True)
    test_loader = DataLoader(test_set, batch_size=1)

    writer = SummaryWriter()
    iter_count = 0
    best_acc_dict = {}

    # rnd_indices = torch.randperm(len(train_set))[:10000]
    # rnd_sampler = SubsetRandomSampler(indices=rnd_indices)
    # train_loader = DataLoader(train_set, batch_size=1, sampler=rnd_sampler)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)

    for epoch in range(3):
        print(f"start epoch {epoch}")

        iter_count = train(model, train_loader, iter_count, writer, args)

        acc_dict = test(model, test_loader)        
        for k, v in acc_dict.items():
            writer.add_scalar(f"metric/{k}", v, iter_count)

        for k, v in acc_dict.items():
            if v > best_acc_dict.get(k, -torch.inf):
                best_acc_dict[k] = v
                save_model(model, None, None, v, iter_count, prefix=f"{args.model_prefix}_{k}")

    save_model(model, None, None, 0, iter_count, flag="end", prefix=args.model_prefix)


def eval_loop(args):
    lut_path = Path("./data/VG-regions-dicts-lite.pkl")

    with open(lut_path, "rb") as f:
        look_up_tables = pickle.load(f)
    
    token_to_idx = look_up_tables["token_to_idx"]

    params_path = Path(args.params_path)    
    model = load_model(
        params_path / "config.json",
        params_path / (args.model_name + ".pth.tar"),
        return_features=False,
        losses=args.losses,
    )
    model.name = "_".join(args.losses)
    model.token_to_idx = token_to_idx

    model.toDevice(device)

    test_set = SnareDataset(mode="valid", filter_visual=True)

    test_loader = DataLoader(test_set, batch_size=1)

    # rnd_indices = torch.randperm(len(test_set))[:10]
    # rnd_sampler = SubsetRandomSampler(indices=rnd_indices)
    # test_loader = DataLoader(test_set, batch_size=1, sampler=rnd_sampler)

    if args.test_view:
        test_view_prediction(model, test_loader, iterations=args.test_iterations)
    else:        
        acc_dict = test(model, test_loader)
        print(acc_dict)
        print(",".join([str((v.item() if torch.is_tensor(v) else v)) for v in acc_dict.values()]))

def main():
    args = get_args()
    if args.train:
        train_loop(args)
    else:
        eval_loop(args)


if __name__ == "__main__":
    main()
