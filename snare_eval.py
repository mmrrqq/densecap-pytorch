import argparse
import csv
from pathlib import Path
import pickle
import time
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
from utils.model_io import load_model, save_model
from model.densecap import DenseCapModel
from utils.snare_dataset import SnareDataset

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# device = torch.device("cpu")
REGION_VIEW_HEAD_LR = 1e-3
CAP_LR = 1e-5
LR = 1e-4
WEIGHT_DECAY = 0
ACCUMULATE_BATCH_SIZE = 64


def get_args():
    """Build argparser for training and evaluation configuration."""
    parser = argparse.ArgumentParser()    

    parser.add_argument("--alternating", action="store_true", default=False, help="Train whole network for the first epoch half, only region view prediction for the second half.")
    parser.add_argument("--config-path", default="model_params", help="Path of a folder containing the models config.json file")
    parser.add_argument("--epochs", default=3, help="Max epochs to train.")
    parser.add_argument("--lookup-tables-path", default="./data/VG-regions-dicts-lite.pkl", help="Path to the pickled look up tables for tokenization created while preprocessing VG.")
    parser.add_argument(
        "--losses",
        nargs="+",
        default=["v", "mv", "vc", "dcs", "mv_dcs", "cvp"], help="Specify the losses to be included for training. Default value contains all possible losses."
    )
    parser.add_argument("--model-name", default="Name of the model checkpoint file (excluding '.pth.tar')")
    parser.add_argument("--model-prefix", default=str(time.time()).split(".")[0], help="Prefix used for model checkpoint saves. Defaults to current timestamp.")
    parser.add_argument("--params-path", default="model_params", help="Path to the model checkpoint folder.")
    parser.add_argument("--snare-annotations-path", default="../snare/amt/folds_adversarial", help="Path to the SNARE annotation files.")
    parser.add_argument("--snare-screenshots-path", default="../snare/data/screenshots", help="Path to the SNARE/ShapeNetSem model screenshots.")
    parser.add_argument("--test-categories", action="store_true", default=False, help="Compare the models performance on SNARE object categories.")
    parser.add_argument("--test-view-iterations", default=10, help="Iterations to test when '--test-view' is specified.")
    parser.add_argument("--test-view", action="store_true", default=False, help="Test random vs. predicted vantage point performance.")
    parser.add_argument("--train", action="store_true", default=False, help="Train/Finetune the model on the SNARE training dataset")


    return parser.parse_args()


def fine_tune(
    model: DenseCapModel,
    data_loader: DataLoader,
    iter_offset: int = 0,
    writer: Optional[SummaryWriter] = None,
    args = {}
):
    """Fine tune the DenseCapModel using the provided SNARE DataLoader.
    """
    model.train()

    # freeze region proposals
    model.rpn.training = False
    for param in model.rpn.parameters():
        param.requires_grad = False

    for param in model.roi_heads.box_predictor.parameters():
        param.requires_grad = False

    view_ids = torch.arange(8)    

    optimizer = torch.optim.AdamW(
        [
            {
                "params": (
                    para
                    for name, para in model.named_parameters()
                    if para.requires_grad
                    and "box_describer" not in name
                    and "region_view_head" not in name
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
                    for para in model.roi_heads.region_view_head.parameters()
                    if para.requires_grad
                ),
                "lr": REGION_VIEW_HEAD_LR,
                "name": "region_view_head",
            },
            {
                "params": (
                    para
                    for para in model.roi_heads.view_predictor_head.parameters()
                    if para.requires_grad
                ) if not args.alternating else (),
                "lr": REGION_VIEW_HEAD_LR,
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
        lr=REGION_VIEW_HEAD_LR        
    ) if args.alternating else None

    iter_count = iter_offset

    cap_view_pred_mode = False

    accum_dict = {}

    for i, batch in enumerate(tqdm(data_loader)):
        imgs, gt_idxs, _, annotation, is_visual = batch

        if is_visual.sum() < data_loader.batch_size or (gt_idxs < 0).sum() > 0:
            continue

        imgs = [img.squeeze().to(device) for img in imgs]        

        loss, loss_dict, _ = model.query_caption(imgs, annotation, view_ids)        

        if cap_view_pred_mode:
            loss = loss_dict['view_prediction'] / ACCUMULATE_BATCH_SIZE            
        else:
            loss = loss / ACCUMULATE_BATCH_SIZE            

        loss.backward()

        # accumulate each loss..
        for k, v in loss_dict.items():
            if accum_dict.get(k):
                accum_dict[k] += v.item() / ACCUMULATE_BATCH_SIZE
            else:
                accum_dict[k] = v.item() / ACCUMULATE_BATCH_SIZE

        # .. until accumulation batch size or data end is reached
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


def test_model_categories(model: DenseCapModel, data_loader: DataLoader) -> Dict[str, float]:
    """Calculate accuracy metrics for the given model on the SNARE benchmark per ShapeNet object category.

    Returns: Dictionary containing accuracies calculated using different metrics. Indexed by metric name.
    """
    model.eval()
    view_ids = torch.arange(8)

    n = 0    
    category_dict = build_category_dict(metadata_path="./data/shapenet_sem_metadata.csv")
    category_accuracy_dict = {}

    with torch.no_grad():
        for batch in tqdm(data_loader):
            (key1_imgs, key2_imgs), gt_idx, (key1, key2), annotation, is_visual = batch

            if not is_visual or gt_idx < 0:
                continue

            annotation = annotation[0]

            # swap images if key2 is referenced
            if gt_idx > 0:
                key1_imgs, key2_imgs = key2_imgs, key1_imgs

            key1_imgs = [k.squeeze().to(device) for k in key1_imgs]
            _, losses1, _ = model.query_caption(key1_imgs, [annotation], view_ids)

            del key1_imgs

            key2_imgs = [k.squeeze().to(device) for k in key2_imgs]
            _, losses2, _ = model.query_caption(key2_imgs, [annotation], view_ids)
            
            # collect correct assignments
            n += 1

            cat_identifiers = category_dict[key1[0]]
            for cat in cat_identifiers:
                if not category_accuracy_dict.get(cat):
                    category_accuracy_dict[cat] = {
                        "n": 0,
                        "dcs": 0,
                        "cap_mean": 0,
                        "cap_min_per_view_mean": 0
                    }
                category_accuracy_dict[cat]["n"] += 1

                if losses2["dcs"] > losses1["dcs"]:
                    category_accuracy_dict[cat]["dcs"] += 1

                if losses2["cap_mean"] > losses1["cap_mean"]:
                    category_accuracy_dict[cat]["cap_mean"] += 1

                if losses2["cap_min_per_view_mean"] > losses1["cap_min_per_view_mean"]:
                    category_accuracy_dict[cat]["cap_min_per_view_mean"] += 1

    for cat, cat_dict in category_accuracy_dict.items():
        for metric_name, value in list(cat_dict.items()):
            if metric_name == "n":
                continue
            
            cat_dict[metric_name] = value / cat_dict["n"]

    print(category_accuracy_dict)    


def test(model: DenseCapModel, data_loader: DataLoader) -> Dict[str, float]:
    """Calculate accuracy metrics for the given model on the SNARE benchmark validation fold.

    Returns: Dictionary containing accuracies calculated using different metrics. Indexed by metric name.
    """
    model.eval()
    view_ids = torch.arange(8)

    n = 0
    min_pos = 0    
    mean_pos = 0    
    min_per_view_mean_pos = 0
    view_pred_correct = 0
    view_pred_total = 0
    cap_view_pred_correct = 0

    cap_view_dict = { i.item(): 0 for i in view_ids }    
    min_cap_view_dict = { i.item(): 0 for i in view_ids }

    with torch.no_grad():
        for batch in tqdm(data_loader):
            (key1_imgs, key2_imgs), gt_idx, (key1, key2), annotation, is_visual = batch

            if not is_visual or gt_idx < 0:
                continue

            annotation = annotation[0]

            # swap images if key2 is referenced
            if gt_idx > 0:
                key1_imgs, key2_imgs = key2_imgs, key1_imgs

            key1_imgs = [k.squeeze().to(device) for k in key1_imgs]
            _, losses1, _ = model.query_caption(key1_imgs, [annotation], view_ids)

            del key1_imgs

            key2_imgs = [k.squeeze().to(device) for k in key2_imgs]
            _, losses2, _ = model.query_caption(key2_imgs, [annotation], view_ids)
            
            # collect correct assignments
            n += 1
            if losses2["dcs"] > losses1["dcs"]:
                min_pos += 1

            if losses2["cap_mean"] > losses1["cap_mean"]:
                mean_pos += 1            

            if losses2["cap_min_per_view_mean"] > losses1["cap_min_per_view_mean"]:
                min_per_view_mean_pos += 1            

            view_preds = losses1["view_preds"].cpu()
            view_pred_total += len(view_preds)
            view_pred_correct += (view_preds == view_ids).sum()
            print(view_pred_correct)     
            print(view_pred_correct / view_pred_total)       
            cap_view_pred_correct += (losses1["cap_min_view"] == losses1["view_cap_preds"]).sum()
            cap_view_dict[losses1["view_cap_preds"].item()] += 1
            min_cap_view_dict[losses1["cap_min_view"].item()] += 1

    mean_acc = mean_pos / n    
    min_acc = min_pos / n
    min_per_view_mean_acc = min_per_view_mean_pos / n    
    view_pred_acc = view_pred_correct / view_pred_total
    cap_view_pred_acc = cap_view_pred_correct / n

    print(
        f"test end.\nmin:\t{min_acc:.2f}\nmean:\t{mean_acc:.2f}\nmin per view mean:\t{min_per_view_mean_acc}\nview pred:\t{view_pred_acc}\ncap view pred:\t{cap_view_pred_acc}"
    )
    print(cap_view_dict)
    print(min_cap_view_dict)
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
    _, best_view_losses, (best_view_caption, _) = model.query_caption([pred_best_img], [query], torch.tensor([best_view_id]))
    _, random_view_losses, (random_view_caption, _) = model.query_caption([rnd_img], [query], torch.tensor([rnd_view_id]))    

    return best_view_losses, random_view_losses, (rnd_view_id, random_view_caption, best_view_caption)


def print_decode_caption(cap: torch.Tensor, idx_to_token):
    for i in cap:
        if i < 1:
            break
        print(idx_to_token[i.item()], end=" ")
    
    print("\n")


def build_category_dict(metadata_path: str = "./data/shapenet_sem_metadata.csv"):
    """Construct a dictionary indexing the object categories by the shapenet model id."""    
    identifier_dict = {}    

    with open(metadata_path) as f:
        csv_reader = csv.reader(f, delimiter=',')
        
        for row in csv_reader:
            identifier, category, *_ = row        
            categories = category.split(",")        

            filtered_categories = [cat for cat in categories if not "_" in cat]
            identifier_dict[identifier] = filtered_categories

    return identifier_dict


def test_view_prediction(model: DenseCapModel, data_loader: DataLoader, iterations=10):
    """Test the caption view prediction performance of the model using the SNARE validation fold for :iterations.
    Calculate accuracy using random vs predicted vantage point and print results.    
    """
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
                if not ("back" in annotation.lower().split(" ")):
                    continue

                if gt_idx > 0:
                    key1, key2 = key2, key1
                    key1_imgs, key2_imgs = key2_imgs, key1_imgs

                key1_imgs = [k.squeeze().to(device) for k in key1_imgs]
                best_view_id = model.query_view_caption(annotation).cpu().item()                
                
                best_view_losses1, random_losses1, (rnd_view_id1, rnd_view_caption1, best_view_caption1) = view_predict(model, key1_imgs, annotation, best_view_id)                
                
                del key1_imgs
                
                key2_imgs = [k.squeeze().to(device) for k in key2_imgs]
                best_view_losses2, random_losses2, (rnd_view_id2, rnd_view_caption2, best_view_caption2) = view_predict(model, key2_imgs, annotation, best_view_id)
                
                n += 1
                if best_view_losses2["dcs"] > best_view_losses1["dcs"]:
                    pred_pos += 1

                if random_losses2["dcs"] > random_losses1["dcs"]:
                    random_pos += 1

                best_view_pred1 = best_view_losses1['view_preds'].cpu().item()
                best_view_pred2 = best_view_losses2['view_preds'].cpu().item()
                random_view_pred1 = random_losses1['view_preds'].cpu().item()
                random_view_pred2 = random_losses2['view_preds'].cpu().item()
                # if rnd_view_id1 == random_view_pred1 and best_view_pred1 == best_view_id and not random_losses2["dcs"] > random_losses1["dcs"] and best_view_losses2["dcs"] > best_view_losses1["dcs"]:
                print(f"{annotation} view: {rnd_view_id1} (pred: {random_view_pred1}) best view: {best_view_id} (pred {best_view_pred1}); id: {key1} ")                    
                print(f"distractor id: {key2}")
                print_decode_caption(rnd_view_caption1, model.idx_to_token)
                print_decode_caption(best_view_caption1, model.idx_to_token)                    

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


def fine_tune_loop(args):
    """Setup and loop the fine tuning.
    """
    print(args.losses)    

    with open(args.lookup_tables_path, "rb") as f:
        look_up_tables = pickle.load(f)
    
    token_to_idx = look_up_tables["token_to_idx"]
    idx_to_token = look_up_tables["idx_to_token"]

    params_path = Path(args.params_path)    
    model = load_model(
        params_path / args.model_name / "config.json",
        params_path / (args.model_name + ".pth.tar"),
        return_features=False,
        losses=args.losses,
    )
    model.name = "_".join(args.losses)
    model.token_to_idx = token_to_idx
    model.idx_to_token = idx_to_token

    model.toDevice(device)
    test_set = SnareDataset(args.snare_annotations_path, args.snare_screenshots_path, mode="valid", filter_visual=True)

    train_set = SnareDataset(args.snare_annotations_path, args.snare_screenshots_path, mode="train", filter_visual=True)
    test_loader = DataLoader(test_set, batch_size=1)

    writer = SummaryWriter()
    iter_count = 0
    best_acc_dict = {}
    
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)

    for epoch in range(args.epochs):
        print(f"start epoch {epoch}")

        iter_count = fine_tune(model, train_loader, iter_count, writer, args)

        acc_dict = test(model, test_loader)        
        for k, v in acc_dict.items():
            writer.add_scalar(f"metric/{k}", v, iter_count)

        for k, v in acc_dict.items():
            if v > best_acc_dict.get(k, -torch.inf):
                best_acc_dict[k] = v
                save_model(model, None, None, v, iter_count, prefix=f"{args.model_prefix}_{k}")

    save_model(model, None, None, 0, iter_count, flag="end", prefix=args.model_prefix)


def eval_model(args):
    with open(args.lookup_tables_path, "rb") as f:
        look_up_tables = pickle.load(f)
    
    token_to_idx = look_up_tables["token_to_idx"]
    idx_to_token = look_up_tables["idx_to_token"]

    params_path = Path(args.params_path)        
    model = load_model(
        params_path / "config.json",
        params_path / (args.model_name + ".pth.tar"),
        return_features=False,
        losses=args.losses,
    )
    model.name = "_".join(args.losses)
    model.token_to_idx = token_to_idx    
    model.idx_to_token = idx_to_token

    model.toDevice(device)

    test_set = SnareDataset(args.snare_annotations_path, args.snare_screenshots_path, mode="valid", filter_visual=True)
    test_loader = DataLoader(test_set, batch_size=1)    

    if args.test_view:
        test_view_prediction(model, test_loader, iterations=args.test_view_iterations)
    elif args.test_categories:
        test_model_categories(model, test_loader)
    else:
        acc_dict = test(model, test_loader)
        print(acc_dict)
        print(",".join([str((v.item() if torch.is_tensor(v) else v)) for v in acc_dict.values()]))

def main():
    args = get_args()

    if args.model_name is None:
        exit("please specify a model name")

    if args.train:
        fine_tune_loop(args)
    else:
        eval_model(args)


if __name__ == "__main__":
    main()
