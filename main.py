import argparse
import os
import random
import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import utils
from networks.hybrid_seg_modeling import QuantumUNet
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from torchinfo import summary
from sklearn.metrics import jaccard_score, accuracy_score

# =========================
# Dataset (S1+S2 → 15 channels)
# =========================
class PatchPairDataset(Dataset):
    def __init__(self, s1_fns, s2_fns,
                 lr_label_fns=None,
                 hr_label_fns=None,
                 image_transform=None,
                 label_transform=None):
        
        assert len(s1_fns) == len(s2_fns), "S1 and S2 lists must match"

        self.s1_fns = list(s1_fns)
        self.s2_fns = list(s2_fns)

        # LR labels (pseudo or LC17)
        self.lr_label_fns = list(lr_label_fns) if lr_label_fns is not None else [None] * len(s1_fns)

        # HR labels (GT)
        self.hr_label_fns = list(hr_label_fns) if hr_label_fns is not None else [None] * len(s1_fns)

        self.image_transform = image_transform
        self.label_transform = label_transform

    def __len__(self):
        return len(self.s1_fns)

    def _read_img(self, fn):
        with rasterio.open(fn, "r") as fp:
            img = fp.read()
        return np.moveaxis(img, 0, 2)

    def _read_label(self, fn):
        if fn is None:
            return None
        with rasterio.open(fn, "r") as fp:
            lab = fp.read(1)
        return lab

    def __getitem__(self, i):
        # --- Load S1 + S2 ---
        s1 = self._read_img(self.s1_fns[i])
        s2 = self._read_img(self.s2_fns[i])
        img = np.concatenate([s1, s2], axis=-1).astype(np.float32)

        img_raw = torch.from_numpy(img).permute(2, 0, 1)
        img_norm = torch.from_numpy(
            ((img - utils.IMAGE_MEANS) / utils.IMAGE_STDS).astype(np.float32)
        ).permute(2, 0, 1)

        # --- Load HR label if available ---
        hr_lab = self._read_label(self.hr_label_fns[i])

        if hr_lab is not None:
            # HR labels are already DFC10 → DO NOT REMAP
            lab_t = torch.from_numpy(hr_lab.astype(np.int64))
            return img_norm, lab_t, img_raw

        # --- Otherwise load LR label (training/pseudo) ---
        lr_lab = self._read_label(self.lr_label_fns[i])

        if lr_lab is not None and self.label_transform is not None:
            # LR labels must be remapped + savanna masked
            lab_t = self.label_transform(lr_lab)
            return img_norm, lab_t, img_raw

        # No labels (rare case)
        return img_norm, img_raw


# =========================
# Losses
# =========================
def weighted_dice_loss(inputs, targets, ignore_index=0, eps=1e-6):
    C = inputs.shape[1]
    probs = F.softmax(inputs, dim=1)
    valid = (targets != ignore_index)
    if valid.sum() == 0:
        return inputs.new_tensor(0.0)

    onehot = F.one_hot(targets.clamp(min=0), num_classes=C).permute(0, 3, 1, 2).float()
    probs = probs * valid.unsqueeze(1)
    onehot = onehot * valid.unsqueeze(1)

    intersection = torch.sum(probs * onehot, (0, 2, 3))
    cardinality  = torch.sum(probs + onehot, (0, 2, 3))
    present = cardinality > 0
    if present.sum() == 0:
        return inputs.new_tensor(0.0)

    dice = (2.0 * intersection[present] + eps) / (cardinality[present] + eps)
    return (1.0 - dice).mean()


class HybridSegLoss(nn.Module):
    def __init__(self, ce_weight=0.3, dice_weight=0.7, ignore_index=0, ce_label_smoothing=0.0):
        super().__init__()
        try:
            self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, label_smoothing=ce_label_smoothing)
        except TypeError:
            self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        inputs = torch.nan_to_num(inputs, nan=0.0, posinf=1e4, neginf=-1e4)
        targets = torch.clamp(targets, 0, inputs.shape[1]-1).long()
        ce_val = self.ce_loss(inputs, targets)
        dice_val = weighted_dice_loss(inputs, targets, ignore_index=self.ignore_index)
        return self.ce_weight * ce_val + self.dice_weight * dice_val

# === SSIM helper (channel-wise, works on multi-band tensors in [0,1]) ===
import torch.nn.functional as _F

def _gaussian_window(ws, sigma, C, device):
    x = torch.arange(ws, dtype=torch.float32, device=device) - ws // 2
    g1 = torch.exp(-(x**2) / (2 * sigma**2))
    g1 = (g1 / g1.sum()).unsqueeze(0)
    w2 = (g1.t() @ g1).unsqueeze(0).unsqueeze(0)           # 1x1xHxW
    return w2.repeat(C, 1, 1, 1)                           # Cx1xHxW

def ssim_torch(x, y, window_size=11, sigma=1.5, C1=0.01**2, C2=0.03**2):
    # x,y: BxCxHxW in [0,1]
    C = x.size(1)
    w = _gaussian_window(window_size, sigma, C, x.device)
    mu_x = _F.conv2d(x, w, padding=window_size//2, groups=C)
    mu_y = _F.conv2d(y, w, padding=window_size//2, groups=C)
    mu_x2, mu_y2, mu_xy = mu_x*mu_x, mu_y*mu_y, mu_x*mu_y
    sigma_x2 = _F.conv2d(x*x, w, padding=window_size//2, groups=C) - mu_x2
    sigma_y2 = _F.conv2d(y*y, w, padding=window_size//2, groups=C) - mu_y2
    sigma_xy = _F.conv2d(x*y, w, padding=window_size//2, groups=C) - mu_xy
    num  = (2*mu_xy + C1) * (2*sigma_xy + C2)
    den  = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    return (num / (den + 1e-12)).mean()

# =========================
# Training phases
# =========================
def save_recon_grid(imgs_raw, recon, epoch, save_dir, nrow=4):
    raw = imgs_raw.detach().cpu().numpy()
    rec = recon.detach().cpu().numpy()
    B = min(nrow, raw.shape[0])

    # RGB indices for S2 inside 15-channel S12 stack
    idx_rgb = [6, 5, 4]  # (B4, B3, B2)

    fig, axs = plt.subplots(2, B, figsize=(3*B, 6))
    for i in range(B):
        raw_rgb = np.transpose(raw[i], (1,2,0))[:, :, idx_rgb]
        rec_rgb = np.transpose(rec[i], (1,2,0))[:, :, idx_rgb]

        axs[0, i].imshow(np.clip(raw_rgb, 0, 1))
        axs[0, i].set_title("Raw RGB")
        axs[0, i].axis("off")

        axs[1, i].imshow(np.clip(rec_rgb, 0, 1))
        axs[1, i].set_title("Recon RGB")
        axs[1, i].axis("off")

    os.makedirs(save_dir, exist_ok=True)
    out_fn = os.path.join(save_dir, f"recon_grid_epoch_{epoch}.png")
    plt.tight_layout()
    plt.savefig(out_fn, dpi=150)
    plt.close(fig)
    print(f"Saved recon grid {out_fn}")

# =========================
# Image Reconstruction
# =========================

def train_recon(args, model, device):
    df_train = pd.read_csv(args.train_list)

    train_dataset = PatchPairDataset(
        s1_fns=df_train["S1image_fn"].values,
        s2_fns=df_train["S2image_fn"].values,
        lr_label_fns=df_train["label_fn"].values,
        image_transform=lambda x: torch.from_numpy(
            ((x - utils.IMAGE_MEANS) / utils.IMAGE_STDS).astype(np.float32)
        ).permute(2,0,1),
        label_transform=lambda y: torch.from_numpy(
            utils.mask_savanna(np.take(utils.LABEL_CLASS_TO_IDX_MAP, y, mode="clip")).astype(np.int64)
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True
    )

    l1 = nn.L1Loss()
    try:
        aux_ce = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=args.aux_label_smoothing)
    except TypeError:
        aux_ce = nn.CrossEntropyLoss(ignore_index=0)

    optimizer = optim.Adam(model.parameters(), lr=args.base_lr)

    model.train()

    for epoch in range(args.epochs_recon):

        # freeze recon head after N epochs
        if epoch == args.freeze_recon_epoch:
            for p in model.recon_conv.parameters():
                p.requires_grad = False
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.base_lr
            )

        recon_w = args.recon_weight if epoch < args.freeze_recon_epoch else 0.0
        aux_w   = args.aux_weight if epoch < args.freeze_recon_epoch else args.aux_weight_after_freeze

        recon_losses, aux_losses, total_losses = [], [], []
        pbar = tqdm(train_loader, desc=f"Recon Epoch {epoch}", leave=True)

        for batch in pbar:
            imgs_norm, lr_labels, imgs_raw = batch
            imgs_norm = imgs_norm.to(device)
            imgs_raw  = imgs_raw.to(device).float()

            # normalize raw to [0,1]
            imgs_raw = torch.clamp(
                (imgs_raw - imgs_raw.amin(dim=(2,3), keepdim=True)) /
                (imgs_raw.amax(dim=(2,3), keepdim=True) - imgs_raw.amin(dim=(2,3), keepdim=True) + 1e-6),
                0, 1
            )

            lr_labels = lr_labels.to(device)

            seg_logits, recon, aux_logits = model(imgs_norm)
            recon_pred = torch.sigmoid(recon)

            loss_l1   = l1(recon_pred, imgs_raw)
            loss_ssim = 1 - ssim_torch(recon_pred, imgs_raw)
            loss_recon = 0.8 * loss_l1 + 0.2 * loss_ssim

            loss_aux = aux_ce(aux_logits, lr_labels)

            loss = recon_w * loss_recon + aux_w * loss_aux

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            recon_losses.append(loss_recon.item())
            aux_losses.append(loss_aux.item())
            total_losses.append(loss.item())

            pbar.set_postfix({
                "recon": f"{np.mean(recon_losses):.4f}",
                "auxCE": f"{np.mean(aux_losses):.4f}",
                "total": f"{np.mean(total_losses):.4f}"
            })

        # save checkpoint
        ckpt = os.path.join(args.savepath, f"recon_epoch_{epoch}.pth")
        torch.save(model.state_dict(), ckpt)

        # save sample recon grid
        with torch.no_grad():
            sample_recon = torch.sigmoid(recon)
            save_recon_grid(imgs_raw, sample_recon, epoch,
                            os.path.join(args.savepath, "recon_samples"))


# =========================
# Pseudo Label Generation
# =========================
def generate_pseudo_labels(args, model, device):
    # Use ONLY the training split for pseudo-label generation
    df = pd.read_csv(args.train_list)

    out_dir = os.path.join(args.savepath, "pseudo_labels")
    os.makedirs(out_dir, exist_ok=True)

    model.eval()
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating pseudo", leave=True):

            s1_fn, s2_fn, lr_label_fn = row["S1image_fn"], row["S2image_fn"], row["label_fn"]

            # --- Load S1 + S2 ---
            with rasterio.open(s1_fn) as s1_fp:
                s1 = np.moveaxis(s1_fp.read(), 0, 2).astype(np.float32)
                H, W = s1_fp.height, s1_fp.width
                profile_hr = s1_fp.profile
                transform_hr, crs_hr = s1_fp.transform, s1_fp.crs

            with rasterio.open(s2_fn) as s2_fp:
                s2 = np.moveaxis(s2_fp.read(), 0, 2).astype(np.float32)
                assert s2.shape[:2] == (H, W)

            img = np.concatenate([s1, s2], axis=-1)
            img_norm = (img - utils.IMAGE_MEANS) / utils.IMAGE_STDS
            img_norm = torch.from_numpy(np.moveaxis(img_norm, -1, 0)).float().unsqueeze(0).to(device)

            # --- Model forward ---
            _, _, aux_logits = model(img_norm)
            probs = F.softmax(aux_logits / args.aux_temp, dim=1)
            conf, pseudo = torch.max(probs, dim=1)
            conf, pseudo = conf.squeeze(0), pseudo.squeeze(0)

            # --- Load LR label ---
            with rasterio.open(lr_label_fn) as lr_fp:
                lr = lr_fp.read(1)
                transform_lr, crs_lr = lr_fp.transform, lr_fp.crs

            # Reproject LR → HR grid if needed
            if lr.shape == (H, W) and transform_lr == transform_hr and crs_lr == crs_hr:
                lr_up = lr
            else:
                lr_up = np.zeros((H, W), dtype=np.uint8)
                rasterio.warp.reproject(
                    source=lr,
                    destination=lr_up,
                    src_transform=transform_lr,
                    src_crs=crs_lr,
                    dst_transform=transform_hr,
                    dst_crs=crs_hr,
                    resampling=rasterio.warp.Resampling.nearest,
                )

            # Remap LR → DFC10
            lr_up = np.take(utils.LABEL_CLASS_TO_IDX_MAP, lr_up, mode="clip").astype(np.int64)
            lr_up = utils.mask_savanna(lr_up)  # ignore savanna
            lr_valid = torch.from_numpy((lr_up != 0).astype(np.uint8)).to(device)

            # --- Confidence filtering ---
            valid_mask = (lr_valid > 0)

            if args.pseudo_keep > 0:
                v = conf[valid_mask].float()
                if v.numel() == 0:
                    thr = torch.tensor(1.0, device=conf.device)
                else:
                    if v.numel() > 2_000_000:
                        v = v[torch.randperm(v.numel(), device=v.device)[:2_000_000]]
                    thr = torch.quantile(v, max(0.0, 1.0 - args.pseudo_keep))
                mask = (conf >= thr) & valid_mask
            else:
                mask = (conf > args.pseudo_thresh) & valid_mask

            pseudo_np = torch.where(mask, pseudo, torch.zeros_like(pseudo)).cpu().numpy()

            # --- Optional morphology ---
            if args.morph_open > 0:
                try:
                    from scipy.ndimage import binary_opening, binary_closing, generate_binary_structure
                    st = generate_binary_structure(2, 1)
                    for cls in range(1, args.num_classes):
                        m = (pseudo_np == cls).astype(np.uint8)
                        if m.sum() == 0:
                            continue
                        m = binary_opening(m, structure=st, iterations=args.morph_open)
                        m = binary_closing(m, structure=st, iterations=max(1, args.morph_open // 2))
                        pseudo_np[(pseudo_np == cls) & (m == 0)] = 0
                except Exception as e:
                    print(f"  Morphology skipped: {e}")

            # --- Save pseudo-label ---
            out_fn = os.path.join(out_dir, os.path.basename(s2_fn).replace(".tif", "_pseudo.tif"))
            prof = profile_hr.copy()
            prof.update(driver="GTiff", dtype="uint8", count=1, nodata=0)

            with rasterio.open(out_fn, "w", **prof) as dst:
                dst.write(pseudo_np.astype(np.uint8), 1)
                dst.write_colormap(1, utils.LABEL_IDX_COLORMAP)


# =========================
# Phase 3: Train segmentation head (no validation)
# =========================
def train_seg(args, model, device):
    pseudo_dir = os.path.join(args.savepath, "pseudo_labels")

    df_train = pd.read_csv(args.train_list)
    df_val   = pd.read_csv(args.val_list)

    # -----------------------------
    # TRAIN SET → use pseudo labels
    # -----------------------------
    df_train["label_fn"] = df_train["S2image_fn"].apply(
        lambda fn: os.path.join(pseudo_dir, os.path.basename(fn).replace(".tif", "_pseudo.tif"))
    )

    train_dataset = PatchPairDataset(
        s1_fns=df_train["S1image_fn"].values,
        s2_fns=df_train["S2image_fn"].values,
        lr_label_fns=df_train["label_fn"].values,
        hr_label_fns=None,
        image_transform=lambda x: torch.from_numpy(
            ((x - utils.IMAGE_MEANS) / utils.IMAGE_STDS).astype(np.float32)
        ).permute(2, 0, 1),
        label_transform=lambda y: torch.from_numpy(
            utils.mask_savanna(
                np.take(utils.LABEL_CLASS_TO_IDX_MAP_GT, y, mode="clip")
            ).astype(np.int64)
        ),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
    )

    # -----------------------------------------
    # VALIDATION SET → use HR labels (GT)
    # -----------------------------------------   
    val_dataset = PatchPairDataset(
        s1_fns=df_val["S1image_fn"].values,
        s2_fns=df_val["S2image_fn"].values,
        lr_label_fns=None,
        hr_label_fns=df_val["hr_label_fn"].values,   # <-- HR labels        

        image_transform=lambda x: torch.from_numpy(
                ((x - utils.IMAGE_MEANS) / utils.IMAGE_STDS).astype(np.float32)
            ).permute(2, 0, 1),
        label_transform=lambda y: torch.from_numpy(
                np.take(utils.LABEL_CLASS_TO_IDX_MAP_GT, y, mode="clip").astype(np.int64)
            ), 
     )   
    
    

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=True,
    )

    # -----------------------------
    # Loss + optimizer
    # -----------------------------
    seg_loss_fn = HybridSegLoss(
        ce_weight=args.ce_weight,
        dice_weight=args.dice_weight,
        ignore_index=0,
        ce_label_smoothing=args.ce_label_smoothing,
    )

    optimizer = optim.Adam(model.parameters(), lr=args.base_lr)

    # -----------------------------
    # Training loop
    # -----------------------------
    for epoch in range(args.max_epochs):
        model.train()
        seg_losses = []

        pbar = tqdm(train_loader, desc=f"Seg Epoch {epoch}", leave=True)
        for imgs_norm, labs, _ in pbar:
            imgs_norm, labs = imgs_norm.to(device), labs.to(device).long()
            labs = labs.clamp_(0, args.num_classes - 1)

            if (labs != 0).sum() == 0:
                continue

            seg_logits, _, _ = model(imgs_norm)
            loss_seg = seg_loss_fn(seg_logits, labs)

            optimizer.zero_grad(set_to_none=True)
            loss_seg.backward()
            optimizer.step()

            seg_losses.append(loss_seg.item())
            pbar.set_postfix({"seg_loss": f"{np.mean(seg_losses):.4f}"})

        # -----------------------------
        # Validation (separate function)
        # -----------------------------
        miou = validate_seg(
            model=model,
            val_loader=val_loader,
            device=device,
            num_classes=args.num_classes,
            epoch=epoch,
            save_dir=args.savepath
        )

        # Save checkpoint
        ckpt = os.path.join(args.savepath, f"seg_epoch_{epoch}.pth")
        torch.save(model.state_dict(), ckpt)
        print(f"Saved checkpoint {ckpt}")

# =========================
# Validation function
# =========================
#@torch.no_grad()
#def validate_seg(model, val_loader, device, num_classes, epoch, save_dir):
#    model.eval()
#    y_true_all, y_pred_all = [], []
#
#    for imgs_norm, labs, _ in tqdm(val_loader, desc="Validating", leave=False):
#        imgs_norm, labs = imgs_norm.to(device), labs.to(device)
#        seg_logits, _, _ = model(imgs_norm)
#        preds = torch.argmax(seg_logits, dim=1)
#
#        valid_mask = (labs != 0)
#        y_true_all.append(labs[valid_mask].cpu().numpy().flatten())
#        y_pred_all.append(preds[valid_mask].cpu().numpy().flatten())
#
#    y_true_all = np.concatenate(y_true_all)
#    y_pred_all = np.concatenate(y_pred_all)

#    oa = accuracy_score(y_true_all, y_pred_all)
#    ious = jaccard_score(y_true_all, y_pred_all, labels=list(range(num_classes)), average=None)
#    miou = np.nanmean(ious)
#
#    print("\n--- Validation Metrics ---")
#    print(f"Epoch {epoch}: OA={oa:.4f}, mIoU={miou:.4f}")
#    for cls_id, val in enumerate(ious):
#        cls_name = utils.LABEL_NAMES.get(cls_id, f"Class {cls_id}")
#        print(f"  {cls_name}: {val:.4f}")
#
#    # Save logs
#    os.makedirs(os.path.join(save_dir, "val_logs"), exist_ok=True)
#    log_path = os.path.join(save_dir, "val_logs", "val_metrics.txt")
#    with open(log_path, "a") as f:
#        f.write(f"Epoch {epoch}: OA={oa:.4f}, mIoU={miou:.4f}\n")
#        for cls_id, val in enumerate(ious):
#            f.write(f"  Class {cls_id}: {val:.4f}\n")
#        f.write("\n")
#
#    return miou
@torch.no_grad()
def validate_seg(model, val_loader, device, num_classes, epoch, save_dir):
    model.eval()
    y_true_all, y_pred_all = [], []

    for imgs_norm, labs, _ in tqdm(val_loader, desc="Validating", leave=False):
        imgs_norm, labs = imgs_norm.to(device), labs.to(device)
        seg_logits, _, _ = model(imgs_norm)
        preds = torch.argmax(seg_logits, dim=1)

        # ignore index = 0
        valid_mask = (labs != 0)
        y_true_all.append(labs[valid_mask].cpu().numpy().flatten())
        y_pred_all.append(preds[valid_mask].cpu().numpy().flatten())

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)

    # -----------------------------
    # 5-class evaluation only
    # -----------------------------
    kept = [1, 4, 6, 7, 10]

    mask_5 = np.isin(y_true_all, kept)
    y_true_5 = y_true_all[mask_5]
    y_pred_5 = y_pred_all[mask_5]

    oa_5 = accuracy_score(y_true_5, y_pred_5)
    ious_5 = jaccard_score(
        y_true_5, y_pred_5,
        labels=kept,
        average=None
    )
    miou_5 = np.nanmean(ious_5)

    print("\n--- Validation Metrics (5-Class WSL) ---")
    print(f"Epoch {epoch}: OA={oa_5:.4f}, mIoU={miou_5:.4f}")
    for cls_id, val in zip(kept, ious_5):
        cls_name = utils.LABEL_NAMES.get(cls_id, f"Class {cls_id}")
        print(f"  {cls_name}: {val:.4f}")

    # Save logs
    os.makedirs(os.path.join(save_dir, "val_logs"), exist_ok=True)
    log_path = os.path.join(save_dir, "val_logs", "val_metrics.txt")

    with open(log_path, "a") as f:
        f.write(f"Epoch {epoch}: OA_5={oa_5:.4f}, mIoU_5={miou_5:.4f}\n")
        for cls_id, val in zip(kept, ious_5):
            f.write(f"  Class {cls_id}: {val:.4f}\n")
        f.write("\n")

    return miou_5


# =========================
# CLI
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="DFC")
    #parser.add_argument("--list_dir", type=str, default = "./dataset/CSV_list/C_NYC-train.csv")#required=True,
                        #help="CSV with columns: S1image_fn,S2image_fn,label_fn")
                        
    parser.add_argument("--train_list", type=str, default="./dataset/CSV_list/C_NYC-train_split.csv")
    parser.add_argument("--val_list", type=str, default="./dataset/CSV_list/C_NYC-val_split.csv")


    parser.add_argument("--max_epochs", type=int, default=31)
    parser.add_argument("--epochs_recon", type=int, default=31)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--base_lr", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--savepath", type=str, default="./log_l2_qunet/dfc/v4/")
    parser.add_argument("--gpu", type=str, default="all")
    parser.add_argument("--num_classes", type=int, default=11, help="0=Ignore + 10 DFC classes")
    parser.add_argument("--num_workers", type=int, default=8)

    # Loss/strategy knobs
    parser.add_argument("--recon_weight", type=float, default=1.0)
    parser.add_argument("--aux_weight", type=float, default=1.0)
    parser.add_argument("--freeze_recon_epoch", type=int, default=41)
    parser.add_argument("--aux_weight_after_freeze", type=float, default=2.0)
    parser.add_argument("--aux_label_smoothing", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=100.0)

    # Seg loss
    parser.add_argument("--ce_weight", type=float, default=0.6)
    parser.add_argument("--dice_weight", type=float, default=0.4)
    parser.add_argument("--ce_label_smoothing", type=float, default=0.02)

    # Pseudo label controls
    parser.add_argument("--pseudo_thresh", type=float, default=0.25)
    parser.add_argument("--pseudo_keep", type=float, default=0.25)
    parser.add_argument("--aux_temp", type=float, default=0.7)
    parser.add_argument("--morph_open", type=int, default=1)

    parser.add_argument("--phase", type=str, choices=["recon", "pseudo", "seg"], required=True)

    args = parser.parse_args()

    if torch.cuda.is_available():
        print(f"Using CUDA device(s): {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(args.seed)

    os.makedirs(args.savepath, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = QuantumUNet(num_classes=args.num_classes, in_channels=15).to(device)
    summary(net)

    if args.phase == "recon":
        train_recon(args, net, device)
    elif args.phase == "pseudo":
        ckpt = os.path.join(args.savepath, f"recon_epoch_{args.epochs_recon-1}.pth")
        net.load_state_dict(torch.load(ckpt, map_location=device))
        summary(net)
        generate_pseudo_labels(args, net, device)
    elif args.phase == "seg":
        ckpt = os.path.join(args.savepath, f"recon_epoch_{args.epochs_recon-1}.pth")
        net.load_state_dict(torch.load(ckpt, map_location=device))
        summary(net)
        train_seg(args, net, device)


if __name__ == "__main__":
    main()
