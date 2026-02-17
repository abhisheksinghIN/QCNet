#!/usr/bin/env python3
"""
evaluate_soa.py
Tile-based inference + metrics on test CSV. Saves GeoTIFF predictions and comparison PNGs.

Example:
python evaluate_soa.py \
    --test_list ./dataset/CSV_list/C_NYC-test.csv \
    --model_path ./runs_from_scratch/unet_best.pth \
    --model unet \
    --save_path ./runs_from_scratch/test_results \
    --chip_size 256 --chip_stride 128
"""
import os
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import utils
from networks.soa import get_model
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score, jaccard_score, classification_report
from torchinfo import summary
from networks.unet_seg_modeling import UNet

# autocast compatibility
try:
    from torch.amp import autocast
    def AUTocast():
        return autocast("cuda")
except Exception:
    from torch.cuda.amp import autocast
    def AUTocast():
        return autocast()

# Tile dataset (same as evaluate.py)
class TileInferenceDataset(Dataset):
    def __init__(self, fn, chip_size, stride, transform=None):
        self.fn = fn
        self.chip_size = chip_size
        self.stride = stride
        self.transform = transform
        with rasterio.open(self.fn) as f:
            self.height, self.width = f.height, f.width
            self.num_channels = f.count

        self.chip_coordinates = [
            (y, x)
            for y in list(range(0, self.height - chip_size, stride)) + [self.height - chip_size]
            for x in list(range(0, self.width - chip_size, stride)) + [self.width - chip_size]
        ]

    def __getitem__(self, idx):
        y, x = self.chip_coordinates[idx]
        with rasterio.open(self.fn) as f:
            img = np.moveaxis(
                f.read(window=Window(x, y, self.chip_size, self.chip_size)), 0, -1
            )
        if self.transform:
            img = self.transform(img)
        return img, np.array((y, x))

    def __len__(self):
        return len(self.chip_coordinates)

def image_transforms(img):
    img = (img - utils.IMAGE_MEANS) / utils.IMAGE_STDS
    img = np.moveaxis(img, -1, 0).astype(np.float32)
    return torch.from_numpy(img)


def inference_and_eval(args, model, device):
    model.eval()
    df = pd.read_csv(args.test_list)
    image_fns = df["image_fn"].values

    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.comparisons_dir, exist_ok=True)
    os.makedirs(args.metrics_log, exist_ok=True)

    y_true_all, y_pred_all = [], []

    for image_idx, image_fn in enumerate(image_fns):
        pred_basename = os.path.basename(image_fn)
        base_name = (pred_basename.replace("_naip-new_pred.tif", "")
                                 .replace("_naip-new.tif", "")
                                 .replace("_pred.tif", "")
                                 .replace(".tif", ""))
        print(f"[{image_idx+1}/{len(image_fns)}] Processing {base_name}...")

        with rasterio.open(image_fn) as src:
            H, W = src.height, src.width
            profile = src.profile.copy()

        dataset = TileInferenceDataset(image_fn, chip_size=args.chip_size, stride=args.chip_stride, transform=image_transforms)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True)

        output_tensor = torch.zeros((args.num_classes, H, W), dtype=torch.float32, device=device)
        count_tensor = torch.zeros((H, W), dtype=torch.float32, device=device)
        kernel = torch.ones((args.chip_size, args.chip_size), dtype=torch.float32, device=device)

        for data, coords in dataloader:
            data = data.to(device, non_blocking=True)
            coords = coords.numpy()
            with torch.no_grad(), AUTocast():
                out = model(data)
                if isinstance(out, (tuple, list)):
                    seg_logits = out[0]
                elif isinstance(out, dict):
                    seg_logits = out.get("logits", list(out.values())[0])
                else:
                    seg_logits = out
                preds_seg = F.softmax(seg_logits, dim=1)

            preds_seg *= kernel.unsqueeze(0).unsqueeze(0)
            for j in range(preds_seg.shape[0]):
                y, x = coords[j]
                output_tensor[:, y:y+args.chip_size, x:x+args.chip_size] += preds_seg[j]
                count_tensor[y:y+args.chip_size, x:x+args.chip_size] += kernel

        output_tensor = output_tensor / count_tensor.clamp(min=1e-6)
        output_hard = output_tensor.argmax(dim=0).byte().cpu().numpy()

        # Save GeoTIFF
        profile.update(driver="GTiff", dtype="uint8", count=1, nodata=0)
        pred_fn = os.path.join(args.save_path, f"{base_name}_naip-new_pred.tif")
        with rasterio.open(pred_fn, "w", **profile) as dst:
            dst.write(output_hard, 1)
            try:
                dst.write_colormap(1, utils.LABEL_IDX_COLORMAP)
            except Exception:
                pass
        print(f"Saved prediction: {pred_fn}")

        # Read GT
        row = df[df["image_fn"].str.contains(os.path.basename(image_fn).replace(".tif",""))]
        gt_fn = None
        if len(row) > 0 and "hr_label_fn" in row.columns:
            gt_fn = row.iloc[0]["hr_label_fn"]
        if gt_fn and os.path.exists(gt_fn):
            with rasterio.open(gt_fn) as src:
                gt = src.read(1)
                gt = np.take(utils.LABEL_CLASS_TO_IDX_MAP_GT, gt, mode="clip")
            valid_mask = (gt != 0)
            y_true_all.append(gt[valid_mask].flatten())
            y_pred_all.append(output_hard[valid_mask].flatten())
        else:
            print(f"[Warn] GT missing for {base_name}. Skipping metrics contribution for this tile.")

        # Read lr for visualization if exists
        lr_rgb = None
        if len(row) > 0 and "label_fn" in row.columns and os.path.exists(row.iloc[0]["label_fn"]):
            with rasterio.open(row.iloc[0]["label_fn"]) as src:
                lr = src.read(1)
                lr = np.take(utils.LABEL_CLASS_TO_IDX_MAP, lr, mode="clip")
                lr_rgb = utils.class_map_to_rgb(lr)

        # HR RGB (search in hr_image_dir)
        hr_rgb = None
        if len(row) > 0 and "image_fn" in row.columns and os.path.exists(row.iloc[0]["image_fn"]):
            with rasterio.open(row.iloc[0]["image_fn"]) as src:
                hr_rgb = np.moveaxis(src.read([1, 2, 3]), 0, -1)
                hr_rgb = np.clip(hr_rgb / 255.0, 0, 1) # added extra
                
                
                
        pred_rgb = utils.class_map_to_rgb(output_hard)
        gt_rgb = utils.class_map_to_rgb(gt) if (gt_fn and os.path.exists(gt_fn)) else np.zeros_like(pred_rgb)

        # Visualization
        fig, axs = plt.subplots(1, 4, figsize=(24, 6))
        axs[0].imshow(hr_rgb if hr_rgb is not None else np.zeros_like(pred_rgb)); axs[0].set_title("HR Image")
        axs[1].imshow(lr_rgb if lr_rgb is not None else np.zeros_like(pred_rgb)); axs[1].set_title("Low-Res Label")
        axs[2].imshow(pred_rgb); axs[2].set_title("Prediction")
        axs[3].imshow(gt_rgb); axs[3].set_title("Ground Truth")
        for ax in axs:
            ax.axis("off")
        patches = [mpatches.Patch(color=np.array(color)/255.0, label=utils.LABEL_NAMES[idx])
                   for idx, color in utils.LABEL_IDX_COLORMAP.items() if idx != 0]
        fig.legend(handles=patches, loc="lower center", ncol=len(patches))
        vis_fn = os.path.join(args.comparisons_dir, f"{base_name}_comparison.png")
        plt.savefig(vis_fn, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved visualization: {vis_fn}")

    # Final metrics
    if len(y_true_all) > 0:
        y_true_all = np.concatenate(y_true_all)
        y_pred_all = np.concatenate(y_pred_all)
        mask = (y_true_all != 0) & (y_true_all != 4)
        y_true_eval = y_true_all[mask]
        y_pred_eval = y_pred_all[mask]
        oa = accuracy_score(y_true_eval, y_pred_eval)
        kappa = cohen_kappa_score(y_true_eval, y_pred_eval)
        miou = jaccard_score(y_true_eval, y_pred_eval, average="macro")
        eval_classes = [1,2,3,5]
        cm = confusion_matrix(y_true_eval, y_pred_eval, labels=eval_classes)
        target_names = [utils.LABEL_NAMES[i] for i in eval_classes]
        report = classification_report(y_true_eval, y_pred_eval, labels=eval_classes, target_names=target_names, digits=4)

        print("\n=== Final Metrics ===")
        print(f"OA: {oa:.4f}, Kappa: {kappa:.4f}, mIoU: {miou:.4f}")
        print("Confusion Matrix:\n", cm)
        print(report)

        with open(os.path.join(args.metrics_log, "metrics.txt"), "w") as f:
            f.write(f"OA: {oa:.4f}\nKappa: {kappa:.4f}\nmIoU: {miou:.4f}\n")
            f.write("Confusion Matrix:\n")
            f.write(np.array2string(cm, separator=', '))
            f.write("\n\nClassification Report:\n")
            f.write(report)
    else:
        print("[Warning] No GT pixels collected; metrics not computed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_list", type=str, default="./dataset/CSV_list/C_NYC-test.csv")
    parser.add_argument("--model_path", type=str, default="./log_l2_qunet/soa/quantumunet/quantumunet_epoch_30.pth")
    parser.add_argument("--model", type=str, default="quantumunet", choices=["unet","deeplabv3","segformer","transunet", "quantumunet"])
    parser.add_argument("--save_path", type=str, default="./log_l2_qunet/soa/quantumunet/test/")
    parser.add_argument("--comparisons_dir", type=str, default="./log_l2_qunet/soa/quantumunet/test/comparisons/")
    parser.add_argument("--metrics_log", type=str, default="./log_l2_qunet/soa/quantumunet/test/")
    parser.add_argument("--hr_image_dir", type=str, default="./dataset/Chesapeake_NewYork_dataset/test/HR_image")
    parser.add_argument("--chip_size", type=int, default=256)
    parser.add_argument("--chip_stride", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_classes", type=int, default=6)
    parser.add_argument("--in_channels", type=int, default=4)
    parser.add_argument("--gpu", type=str, default="0")
    args = parser.parse_args()

    if args.comparisons_dir is None:
        args.comparisons_dir = os.path.join(args.save_path, "comparisons")
    if args.metrics_log is None:
        args.metrics_log = args.save_path
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.comparisons_dir, exist_ok=True)
    os.makedirs(args.metrics_log, exist_ok=True)

    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        device = torch.device("cuda")
        print(f"Using CUDA device(s): {args.gpu}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    model = get_model(args.model, num_classes=args.num_classes, in_channels=args.in_channels, img_size=args.chip_size)
    #state_dict = torch.load(args.model_path, map_location="cpu")
    #model.load_state_dict(state_dict)
    
    
    import torch.nn.functional as F
    from math import sqrt
    
    # load checkpoint
    ckpt = torch.load(args.model_path, map_location="cpu")
    if isinstance(ckpt, dict):
        state_dict = ckpt.get("state_dict", ckpt.get("model", ckpt))
    else:
        state_dict = ckpt
    
    model_dict = model.state_dict()
    
    # --- find pos_embed in checkpoint ---
    pos_key = None
    for k in state_dict.keys():
        if "pos_embed" in k:
            pos_key = k
            break
    
    if pos_key is not None and pos_key in model_dict:
        pe_ckpt = state_dict[pos_key]   # e.g. [1, 1024, 512]
        pe_model = model_dict[pos_key]  # e.g. [1, 256, 512]
    
        if pe_ckpt.shape != pe_model.shape:
            print(f"\nInterpolating pos_embed: {tuple(pe_ckpt.shape)} -> {tuple(pe_model.shape)}")
    
            ntok_old = pe_ckpt.shape[1]
            ntok_new = pe_model.shape[1]
            dim = pe_ckpt.shape[2]
    
            gs_old = int(sqrt(ntok_old))
            gs_new = int(sqrt(ntok_new))
    
            assert gs_old * gs_old == ntok_old, "old pos_embed is not square"
            assert gs_new * gs_new == ntok_new, "new pos_embed is not square"
    
            pe = pe_ckpt.reshape(1, gs_old, gs_old, dim).permute(0,3,1,2)       # (1,dim,H,W)
            pe = F.interpolate(pe, size=(gs_new, gs_new), mode="bilinear", align_corners=False)
            pe = pe.permute(0,2,3,1).reshape(1, ntok_new, dim)
    
            state_dict[pos_key] = pe
            print("Interpolation complete.\n")
    
    # --- load everything, skipping bad keys automatically ---
    filtered = {}
    skipped = []
    for k, v in state_dict.items():
        if k in model_dict and v.shape == model_dict[k].shape:
            filtered[k] = v
        else:
            skipped.append(k)
    
    print("Skipped keys:", skipped)
    
    model_dict.update(filtered)
    model.load_state_dict(model_dict)
    print("? Loaded model with interpolated pos_embed.\n")
    
    
    
    
    
    model.eval()
    summary(model)
    model = model.to(device)
    inference_and_eval(args, model, device)
