#!/usr/bin/env python3
"""
evaluate_soa.py
Patch-based inference + metrics on test CSV (S1+S2, 15 channels).
No tiling needed since patches are already 256x256.
"""

import os
import argparse
import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm

import torch
import torch.nn.functional as F
import utils
from networks.soa import get_model
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score, jaccard_score, classification_report
from torchinfo import summary


def image_transforms(img):
    img = (img - utils.IMAGE_MEANS_val) / utils.IMAGE_STDS_val
    img = np.moveaxis(img, -1, 0).astype(np.float32)
    return torch.from_numpy(img)


def inference_and_eval(args, model, device):
    model.eval()
    df = pd.read_csv(args.test_list)

    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.comparisons_dir, exist_ok=True)
    os.makedirs(args.metrics_log, exist_ok=True)

    y_true_all, y_pred_all = [], []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        s1_fn, s2_fn = row["S1image_fn"], row["S2image_fn"]
        lr_fn = row.get("label_fn", None)
        gt_fn = row.get("hr_label_fn", None)

        base_name = os.path.basename(s2_fn).replace(".tif", "")
        print(f"[{idx+1}/{len(df)}] Processing {base_name}...")

        # Read patch (S1+S2)
        s1 = np.moveaxis(rasterio.open(s1_fn).read(), 0, 2).astype(np.float32)
        s2 = np.moveaxis(rasterio.open(s2_fn).read(), 0, 2).astype(np.float32)
        img = np.concatenate([s1, s2], axis=-1)  # [H,W,15]
        H, W, _ = img.shape

        img_tensor = image_transforms(img).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(img_tensor)
            if isinstance(out, (tuple, list)):
                seg_logits = out[0]
            elif isinstance(out, dict):
                seg_logits = out.get("logits", list(out.values())[0])
            else:
                seg_logits = out
            preds_seg = F.softmax(seg_logits, dim=1)
            output_hard = preds_seg.argmax(dim=1).squeeze(0).byte().cpu().numpy()

        # Save GeoTIFF
        profile = rasterio.open(s2_fn).profile
        profile.update(driver="GTiff", dtype="uint8", count=1, nodata=0)
        pred_fn = os.path.join(args.save_path, f"{base_name}_pred.tif")
        with rasterio.open(pred_fn, "w", **profile) as dst:
            dst.write(output_hard, 1)
            try:
                dst.write_colormap(1, utils.LABEL_IDX_COLORMAP)
            except Exception:
                pass
        print(f"Saved prediction: {pred_fn}")

        # Collect GT for metrics
        gt = None
        if gt_fn and os.path.exists(gt_fn):
            gt = rasterio.open(gt_fn).read(1)
            gt = np.take(utils.LABEL_CLASS_TO_IDX_MAP_GT, gt, mode="clip")
            valid_mask = (gt != 0)
            y_true_all.append(gt[valid_mask].flatten())
            y_pred_all.append(output_hard[valid_mask].flatten())

#        # Visualization
#        # S1 grayscale composite (VV/VH)
#        s1_rgb = np.clip(s1[..., :2], 0, 255)
#        s1_rgb = (s1_rgb - s1_rgb.min()) / (s1_rgb.max() - s1_rgb.min() + 1e-6)
#
#        # S2 RGB (B4, B3, B2)
#        s2_rgb = np.stack([s2[...,2], s2[...,1], s2[...,0]], axis=-1)
#        s2_rgb = np.clip(s2_rgb / 3000.0, 0, 1)
#
#        lr_rgb = None
#        if lr_fn and os.path.exists(lr_fn):
#            lr = rasterio.open(lr_fn).read(1)
#            lr = np.take(utils.LABEL_CLASS_TO_IDX_MAP, lr, mode="clip")
#            lr_rgb = utils.class_map_to_rgb(lr)
#
#        pred_rgb = utils.class_map_to_rgb(output_hard)
#        gt_rgb = utils.class_map_to_rgb(gt) if gt is not None else np.zeros_like(pred_rgb)
#
#        fig, axs = plt.subplots(1, 4, figsize=(24, 6))       
#        # ADDED
#        s1_rgb = np.zeros((H, W, 3), dtype=np.float32)
#        s1_rgb[...,0] = (s1[...,0] - s1[...,0].min()) / (s1[...,0].ptp() + 1e-6)  # VV → Red
#        s1_rgb[...,1] = (s1[...,1] - s1[...,1].min()) / (s1[...,1].ptp() + 1e-6)  # VH → Green
#        # Blue left as 0
#        axs[0].imshow(s1_rgb)
#        axs[0].set_title("S1 (VV/VH composite)")        
#        #axs[0].imshow(s1_rgb); axs[0].set_title("S1 (VV/VH)")
#        axs[1].imshow(s2_rgb); axs[1].set_title("S2 (RGB)")
#        axs[2].imshow(lr_rgb if lr_rgb is not None else np.zeros_like(pred_rgb)); axs[2].set_title("LR Label")
#        axs[3].imshow(gt_rgb); axs[3].set_title("HR Label (GT)")
#        for ax in axs: ax.axis("off")
#        patches = [mpatches.Patch(color=np.array(color)/255.0, label=utils.LABEL_NAMES[idx])
#                   for idx, color in utils.LABEL_IDX_COLORMAP.items() if idx != 0]
#        fig.legend(handles=patches, loc="lower center", ncol=len(patches))
#        vis_fn = os.path.join(args.comparisons_dir, f"{base_name}_comparison.png")
#        plt.savefig(vis_fn, dpi=300, bbox_inches="tight")
#        plt.close()
#        print(f"Saved visualization: {vis_fn}")

        # --- Visualization ---
        # Build RGB composites
        s1_rgb = np.zeros((H, W, 3), dtype=np.float32)
        s1_rgb[...,0] = (s1[...,0] - s1[...,0].min()) / (s1[...,0].ptp() + 1e-6)  # VV → Red
        s1_rgb[...,1] = (s1[...,1] - s1[...,1].min()) / (s1[...,1].ptp() + 1e-6)  # VH → Green
        
        s2_rgb = np.stack([s2[...,2], s2[...,1], s2[...,0]], axis=-1)  # B4,B3,B2
        s2_rgb = np.clip(s2_rgb / 3000.0, 0, 1)
        
        lr_rgb = None
        if lr_fn and os.path.exists(lr_fn):
            lr = rasterio.open(lr_fn).read(1)
            lr = np.take(utils.LABEL_CLASS_TO_IDX_MAP, lr, mode="clip")
            lr_rgb = utils.class_map_to_rgb(lr)
        
        pred_rgb = utils.class_map_to_rgb(output_hard)
        gt_rgb   = utils.class_map_to_rgb(gt) if gt is not None else np.zeros_like(pred_rgb)
        
        # --- Plot 5 panels: S1, S2, LR, Prediction, GT ---
        fig, axs = plt.subplots(1, 5, figsize=(30, 6))
        axs[0].imshow(s1_rgb); axs[0].set_title("S1 (VV/VH)")
        axs[1].imshow(s2_rgb); axs[1].set_title("S2 (RGB)")
        axs[2].imshow(lr_rgb if lr_rgb is not None else np.zeros_like(pred_rgb)); axs[2].set_title("LR Label")
        axs[3].imshow(pred_rgb); axs[3].set_title("Prediction")
        axs[4].imshow(gt_rgb); axs[4].set_title("HR Label (GT)")
        for ax in axs: ax.axis("off")
        
        # Legend
#        patches = [mpatches.Patch(color=np.array(color)/255.0, label=utils.LABEL_NAMES[idx])
#                   for idx, color in utils.LABEL_IDX_COLORMAP.items() if idx != 0]
        kept = [1, 4, 6, 7, 10]
        
        patches = [
            mpatches.Patch(
                color=np.array(utils.LABEL_IDX_COLORMAP[idx]) / 255.0,
                label=utils.LABEL_NAMES[idx]
            )
            for idx in kept
        ]
        fig.legend(handles=patches, loc="lower center", ncol=len(patches))
        
        vis_fn = os.path.join(args.comparisons_dir, f"{base_name}_comparison.png")
        plt.savefig(vis_fn, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved visualization: {vis_fn}")



#    # Final metrics
#    if len(y_true_all) > 0:
#        y_true_all = np.concatenate(y_true_all)
#        y_pred_all = np.concatenate(y_pred_all)
#        mask = (y_true_all != 0)
#        y_true_eval = y_true_all[mask]
#        y_pred_eval = y_pred_all[mask]
#        oa = accuracy_score(y_true_eval, y_pred_eval)
#        kappa = cohen_kappa_score(y_true_eval, y_pred_eval)
#        miou = jaccard_score(y_true_eval, y_pred_eval, average="macro")
#        cm = confusion_matrix(y_true_eval, y_pred_eval, labels=list(range(1, args.num_classes)))
#        report = classification_report(y_true_eval, y_pred_eval, labels=list(range(1, args.num_classes)),
#                                       target_names=[utils.LABEL_NAMES[i] for i in range(1, args.num_classes)], digits=4)
#
#        print("\n=== Final Metrics ===")
#        print(f"OA: {oa:.4f}, Kappa: {kappa:.4f}, mIoU: {miou:.4f}")
#        print("Confusion Matrix:\n", cm)
#        print(report)
#
#        with open(os.path.join(args.metrics_log, "metrics.txt"), "w") as f:
#            f.write(f"OA: {oa:.4f}\nKappa: {kappa:.4f}\nmIoU: {miou:.4f}\n")
#            f.write("Confusion Matrix:\n")
#            f.write(np.array2string(cm, separator=', '))
#            f.write("\n\nClassification Report:\n")
#            f.write(report)
#    else:
#        print("[Warning] No GT pixels collected; metrics not computed.")

    # Final metrics (5-class WSL evaluation)
    if len(y_true_all) > 0:
        y_true_all = np.concatenate(y_true_all)
        y_pred_all = np.concatenate(y_pred_all)
    
        # 5-class WSL evaluation
        kept = [1, 4, 6, 7, 10]
        
        mask = np.isin(y_true_all, kept)
        y_true_eval = y_true_all[mask]
        y_pred_eval = y_pred_all[mask]
        
        oa = accuracy_score(y_true_eval, y_pred_eval)
        kappa = cohen_kappa_score(y_true_eval, y_pred_eval)
        ious = jaccard_score(y_true_eval, y_pred_eval, labels=kept, average=None)
        miou = np.nanmean(ious)
        
        cm = confusion_matrix(y_true_eval, y_pred_eval, labels=kept)
        report = classification_report(
            y_true_eval,
            y_pred_eval,
            labels=kept,
            target_names=[utils.LABEL_NAMES[i] for i in kept],
            digits=4
        )
        
        print("\n=== Final Metrics (5-Class WSL) ===")
        print(f"OA: {oa:.4f}, Kappa: {kappa:.4f}, mIoU: {miou:.4f}")
        print("Confusion Matrix:\n", cm)
        print(report)
        
        with open(os.path.join(args.metrics_log, "metrics.txt"), "w") as f:
            f.write(f"OA: {oa:.4f}\nKappa: {kappa:.4f}\nmIoU: {miou:.4f}\n")
            f.write("Confusion Matrix:\n")
            f.write(np.array2string(cm, separator=', '))
            f.write("\n\nClassification Report:\n")
            f.write(report)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_list", type=str, default="./dataset/CSV_list/C_NYC-test.csv")
    parser.add_argument("--model_path", type=str, default="./log_l2_qunet/dfc/transunet/seg_epoch_30.pth")
    parser.add_argument("--model", type=str, default="transunet",
                        choices=["unet","deeplabv3","segformer","transunet","quantumunet"])
    parser.add_argument("--save_path", type=str, default="./log_l2_qunet/dfc/transunet/test/")
    parser.add_argument("--comparisons_dir", type=str, default="./log_l2_qunet/dfc/transunet/test/comparisons/")
    parser.add_argument("--metrics_log", type=str, default="./log_l2_qunet/dfc/transunet/test/")
    parser.add_argument("--num_classes", type=int, default=11)
    parser.add_argument("--in_channels", type=int, default=15)
    parser.add_argument("--gpu", type=str, default="0")
    args = parser.parse_args()

    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        device = torch.device("cuda")
        print(f"Using CUDA device(s): {args.gpu}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    model = get_model(args.model, num_classes=args.num_classes, in_channels=args.in_channels, img_size=256)
    #ckpt = torch.load(args.model_path, map_location="cpu")
    #state_dict = ckpt.get("state_dict", ckpt)
    #model.load_state_dict(state_dict, strict=False)
    
    #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
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
    #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    
    
    
    model = model.to(device)
    summary(model)

    # run inference + evaluation
    inference_and_eval(args, model, device)