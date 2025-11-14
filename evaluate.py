# evaluate.py
import argparse
import os
import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import pandas as pd
import utils
from networks.hybrid_seg_modeling import QuantumUNet
from sklearn.metrics import (
    accuracy_score, confusion_matrix, cohen_kappa_score,
    jaccard_score, classification_report
)
import matplotlib.patches as mpatches
import matplotlib
matplotlib.use('Agg')  # ? headless backend (no Tkinter)



# -------------------- autocast (PyTorch <2.0 vs >=2.0) -------------------- #
try:
    from torch.amp import autocast  # PyTorch >= 2.0
    def AUTocast():
        return autocast("cuda")
except ImportError:
    from torch.cuda.amp import autocast  # PyTorch < 2.0
    def AUTocast():
        return autocast()


# -------------------- Dataset Config -------------------- #
dataset_config = {
    'Chesapeake': {
        'list_dir': './dataset/CSV_list/C_NYC-test.csv',
        'num_classes': 6
    }
}


# ==== Dataset for tiling inference ====
#class TileInferenceDataset(Dataset):
#    def __init__(self, fn, chip_size, stride, transform=None):
#        self.fn = fn
#        self.chip_size = chip_size
#        self.stride = stride
#        self.transform = transform
#
#        with rasterio.open(self.fn) as f:
#            self.height, self.width = f.height, f.width
#            self.num_channels = f.count
#            self.data = np.moveaxis(f.read(), 0, -1)
#
#        self.chip_coordinates = [
#            (y, x)
#            for y in list(range(0, self.height - chip_size, stride)) + [self.height - chip_size]
#            for x in list(range(0, self.width - chip_size, stride)) + [self.width - chip_size]
#        ]
#
#    def __getitem__(self, idx):
#        y, x = self.chip_coordinates[idx]
#        img = self.data[y:y+self.chip_size, x:x+self.chip_size]
#        if self.transform:
#            img = self.transform(img)
#        return img, np.array((y, x))
#
#    def __len__(self):
#        return len(self.chip_coordinates)
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
                f.read(window=rasterio.windows.Window(x, y, self.chip_size, self.chip_size)), 0, -1
            )
        if self.transform:
            img = self.transform(img)
        return img, np.array((y, x))

    def __len__(self):
        return len(self.chip_coordinates)


# ==== Image transform ====
def image_transforms(img):
    img = (img - utils.IMAGE_MEANS) / utils.IMAGE_STDS
    img = np.moveaxis(img, -1, 0).astype(np.float32)
    return torch.from_numpy(img)


# ==== Inference + Evaluation ====
def inference_and_eval(args, model):
    model.eval()
    input_dataframe = pd.read_csv(args.list_dir)
    image_fns = input_dataframe["image_fn"].values

    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.comparisons_dir, exist_ok=True)

    y_true_all, y_pred_all = [], []

    for image_idx, image_fn in enumerate(image_fns):
        # --- Derive base_name correctly ---
        pred_basename = os.path.basename(image_fn)
        base_name = (
            pred_basename.replace("_naip-new_pred.tif", "")
                         .replace("_naip-new.tif", "")
                         .replace("_pred.tif", "")
                         .replace(".tif", "")
        )

        print(f"[{image_idx+1}/{len(image_fns)}] Processing {base_name}...")

        with rasterio.open(image_fn) as src:
            H, W = src.height, src.width
            profile = src.profile.copy()

        dataset = TileInferenceDataset(
            image_fn,
            chip_size=args.chip_size,
            stride=args.chip_stride,
            transform=image_transforms
        )
        dataloader = DataLoader(dataset, batch_size=128, num_workers=4, pin_memory=True)

        output_tensor = torch.zeros((args.num_classes, H, W), dtype=torch.float32, device=device)
        count_tensor = torch.zeros((H, W), dtype=torch.float32, device=device)
        kernel = torch.ones((args.chip_size, args.chip_size), dtype=torch.float32, device=device)

        for data, coords in dataloader:
            data = data.to(device, non_blocking=True)
            coords = coords.numpy()

            with torch.no_grad(), AUTocast():
#                preds_seg, _ = model(data)
#                preds_seg = F.softmax(preds_seg, dim=1)
                seg_logits, _, _ = model(data)
                preds_seg = F.softmax(seg_logits, dim=1)
                
            preds_seg *= kernel.unsqueeze(0).unsqueeze(0)

            for j in range(preds_seg.shape[0]):
                y, x = coords[j]
                output_tensor[:, y:y+args.chip_size, x:x+args.chip_size] += preds_seg[j]
                count_tensor[y:y+args.chip_size, x:x+args.chip_size] += kernel

        output_tensor = output_tensor / count_tensor.clamp(min=1e-6)
        output_hard = output_tensor.argmax(dim=0).byte().cpu().numpy()

        # Save prediction raster
        profile.update(driver="GTiff", dtype="uint8", count=1, nodata=0)
        pred_fn = os.path.join(args.save_path, f"{base_name}_naip-new_pred.tif")
        with rasterio.open(pred_fn, "w", **profile) as dst:
            dst.write(output_hard, 1)
            dst.write_colormap(1, utils.LABEL_IDX_COLORMAP)
        print(f"Saved prediction: {pred_fn}")

        # === Read ground truth + LR labels ===
        gt_fn = os.path.join(args.hr_label_dir, f"{base_name}_lc.tif")
        lr_fn = os.path.join(args.lr_label_dir, f"{base_name}_nlcd.tif")

        with rasterio.open(gt_fn) as src:
            gt = src.read(1)
            gt = np.take(utils.LABEL_CLASS_TO_IDX_MAP_GT, gt, mode='clip')  # remap HR labels

        valid_mask = (gt != 0)
        y_true_all.append(gt[valid_mask].flatten())
        y_pred_all.append(output_hard[valid_mask].flatten())

        with rasterio.open(lr_fn) as src:
            lr = src.read(1)
            lr = np.take(utils.LABEL_CLASS_TO_IDX_MAP, lr, mode='clip')  # remap LR labels

        # === HR RGB for context ===
        rgb_files = [f for f in os.listdir(args.hr_image_dir) if f.startswith(base_name)]
        hr_rgb = None
        if rgb_files:
            with rasterio.open(os.path.join(args.hr_image_dir, rgb_files[0])) as src:
                hr_rgb = np.moveaxis(src.read([1, 2, 3]), 0, -1)
                hr_rgb = np.clip(hr_rgb / 255.0, 0, 1) # added extra


        pred_rgb = utils.class_map_to_rgb(output_hard)
        gt_rgb   = utils.class_map_to_rgb(gt)
        lr_rgb   = utils.class_map_to_rgb(lr)

        # === Visualization with legend ===
        fig, axs = plt.subplots(1, 4, figsize=(24, 6))
        axs[0].imshow(hr_rgb) if hr_rgb is not None else axs[0].imshow(np.zeros_like(gt_rgb))
        axs[0].set_title("HR Image")
        axs[1].imshow(lr_rgb); axs[1].set_title("Low-Res Label")
        axs[2].imshow(pred_rgb); axs[2].set_title("Prediction")
        axs[3].imshow(gt_rgb); axs[3].set_title("Ground Truth")

        for ax in axs:
            ax.axis("off")

        # Add legend (exclude class 0)
        patches = [
            mpatches.Patch(color=np.array(color)/255.0, label=utils.LABEL_NAMES[idx])
            for idx, color in utils.LABEL_IDX_COLORMAP.items() if idx != 0
        ]
        fig.legend(handles=patches, loc="lower center", ncol=len(patches))

        vis_fn = os.path.join(args.comparisons_dir, f"{base_name}_comparison.png")
        plt.savefig(vis_fn, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved visualization: {vis_fn}")

    # === Final metrics ===
    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)

    # mask out ignore (0) and barren (4)
    mask = (y_true_all != 0) & (y_true_all != 4)
    y_true_eval = y_true_all[mask]
    y_pred_eval = y_pred_all[mask]

    # compute metrics only on valid classes
    oa = accuracy_score(y_true_eval, y_pred_eval)
    kappa = cohen_kappa_score(y_true_eval, y_pred_eval)
    miou = jaccard_score(y_true_eval, y_pred_eval, average="macro")

    eval_classes = [1, 2, 3, 5]
    cm = confusion_matrix(y_true_eval, y_pred_eval, labels=eval_classes)
    target_names = [utils.LABEL_NAMES[i] for i in eval_classes]

    report = classification_report(
        y_true_eval, y_pred_eval,
        labels=eval_classes,
        target_names=target_names,
        digits=4
    )


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


# ==== Main ====
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Chesapeake")
    parser.add_argument("--list_dir", type=str, default=dataset_config["Chesapeake"]["list_dir"])
    parser.add_argument("--num_classes", type=int, default=dataset_config["Chesapeake"]["num_classes"])
    parser.add_argument("--model_path", type=str, default="./log_l2_qunet/test/seg_epoch_30.pth")
    parser.add_argument("--save_path", type=str, default="./log_l2_qunet/test/test/")
    parser.add_argument("--metrics_log", type=str, default="./log_l2_qunet/test/test/")
    parser.add_argument("--comparisons_dir", type=str, default="./log_l2_qunet/test/test/comparisons/")
    parser.add_argument("--hr_label_dir", type=str, default="/home/absingh/downloads/quantum/dataset/Chesapeake_NewYork_dataset/test/HR_label")
    parser.add_argument("--lr_label_dir", type=str, default="/home/absingh/downloads/quantum/dataset/Chesapeake_NewYork_dataset/test/LR_label")
    parser.add_argument("--hr_image_dir", type=str, default="/home/absingh/downloads/quantum/dataset/Chesapeake_NewYork_dataset/test/HR_image")
    parser.add_argument("--chip_size", type=int, default=256)
    parser.add_argument("--chip_stride", type=int, default=128)  # or 512 if you want speed

    parser.add_argument("--gpu", type=str, default="0")
    args = parser.parse_args()

    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        device = torch.device("cuda")
        print(f"Using CUDA device(s): {args.gpu}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    model = QuantumUNet(num_classes=args.num_classes, in_channels=4).to(device)
    state_dict = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(device)

    inference_and_eval(args, model)


## evaluate.py
#import argparse
#import os
#import numpy as np
#import rasterio
#import torch
#import torch.nn.functional as F
#from torch.utils.data import DataLoader, Dataset
#import matplotlib.pyplot as plt
#import pandas as pd
#import utils
#from networks.hybrid_seg_modeling import QuantumUNet
#from sklearn.metrics import (
#    accuracy_score, confusion_matrix, cohen_kappa_score,
#    jaccard_score, classification_report
#)
#import matplotlib.patches as mpatches
#
## -------------------- autocast (PyTorch <2.0 vs >=2.0) -------------------- #
#try:
#    from torch.amp import autocast  # PyTorch >= 2.0
#    def AUTocast():
#        return autocast("cuda")
#except ImportError:
#    from torch.cuda.amp import autocast  # PyTorch < 2.0
#    def AUTocast():
#        return autocast()
#
#
## -------------------- Dataset Config -------------------- #
#dataset_config = {
#    'Chesapeake': {
#        'list_dir': './dataset/CSV_list/C_NYC-test.csv',
#        'num_classes': 6
#    }
#}
#
#
## ==== Dataset for tiling inference ====
#class TileInferenceDataset(Dataset):
#    def __init__(self, fn, chip_size, stride, transform=None):
#        self.fn = fn
#        self.chip_size = chip_size
#        self.stride = stride
#        self.transform = transform
#
#        with rasterio.open(self.fn) as f:
#            self.height, self.width = f.height, f.width
#            self.num_channels = f.count
#            self.data = np.moveaxis(f.read(), 0, -1)
#
#        self.chip_coordinates = [
#            (y, x)
#            for y in list(range(0, self.height - chip_size, stride)) + [self.height - chip_size]
#            for x in list(range(0, self.width - chip_size, stride)) + [self.width - chip_size]
#        ]
#
#    def __getitem__(self, idx):
#        y, x = self.chip_coordinates[idx]
#        img = self.data[y:y+self.chip_size, x:x+self.chip_size]
#        if self.transform:
#            img = self.transform(img)
#        return img, np.array((y, x))
#
#    def __len__(self):
#        return len(self.chip_coordinates)
#
#
## ==== Image transform ====
#def image_transforms(img):
#    img = (img - utils.IMAGE_MEANS) / utils.IMAGE_STDS
#    img = np.moveaxis(img, -1, 0).astype(np.float32)
#    return torch.from_numpy(img)
#
#
## ==== Inference + Evaluation ====
#def inference_and_eval(args, model):
#    model.eval()
#    input_dataframe = pd.read_csv(args.list_dir)
#    image_fns = input_dataframe["image_fn"].values
#
#    os.makedirs(args.save_path, exist_ok=True)
#    os.makedirs(args.comparisons_dir, exist_ok=True)
#
#    y_true_all, y_pred_all = [], []
#
#    for image_idx, image_fn in enumerate(image_fns):
#        # --- Derive base_name correctly ---
#        pred_basename = os.path.basename(image_fn)
#        base_name = (
#            pred_basename.replace("_naip-new_pred.tif", "")
#                         .replace("_naip-new.tif", "")
#                         .replace("_pred.tif", "")
#                         .replace(".tif", "")
#        )
#
#        print(f"[{image_idx+1}/{len(image_fns)}] Processing {base_name}...")
#
#        with rasterio.open(image_fn) as src:
#            H, W = src.height, src.width
#            profile = src.profile.copy()
#
#        dataset = TileInferenceDataset(
#            image_fn,
#            chip_size=args.chip_size,
#            stride=args.chip_stride,
#            transform=image_transforms
#        )
#        dataloader = DataLoader(dataset, batch_size=64, num_workers=4, pin_memory=True)
#
#        output_tensor = torch.zeros((args.num_classes, H, W), dtype=torch.float32, device=device)
#        count_tensor = torch.zeros((H, W), dtype=torch.float32, device=device)
#        kernel = torch.ones((args.chip_size, args.chip_size), dtype=torch.float32, device=device)
#
#        for data, coords in dataloader:
#            data = data.to(device, non_blocking=True)
#            coords = coords.numpy()
#
#            with torch.no_grad(), AUTocast():
#                preds_seg, _ = model(data)
#                preds_seg = F.softmax(preds_seg, dim=1)
#
#            preds_seg *= kernel.unsqueeze(0).unsqueeze(0)
#
#            for j in range(preds_seg.shape[0]):
#                y, x = coords[j]
#                output_tensor[:, y:y+args.chip_size, x:x+args.chip_size] += preds_seg[j]
#                count_tensor[y:y+args.chip_size, x:x+args.chip_size] += kernel
#
#        output_tensor = output_tensor / count_tensor.clamp(min=1e-6)
#        output_hard = output_tensor.argmax(dim=0).byte().cpu().numpy()
#
#        # Save prediction raster
#        profile.update(driver="GTiff", dtype="uint8", count=1, nodata=0)
#        pred_fn = os.path.join(args.save_path, f"{base_name}_naip-new_pred.tif")
#        with rasterio.open(pred_fn, "w", **profile) as dst:
#            dst.write(output_hard, 1)
#            dst.write_colormap(1, utils.LABEL_IDX_COLORMAP)
#        print(f"Saved prediction: {pred_fn}")
#
#        # === Read ground truth + LR labels ===
#        gt_fn = os.path.join(args.hr_label_dir, f"{base_name}_lc.tif")
#        lr_fn = os.path.join(args.lr_label_dir, f"{base_name}_nlcd.tif")
#
#        with rasterio.open(gt_fn) as src:
#            gt = src.read(1)
#            gt = utils.LABEL_CLASS_TO_IDX_MAP_GT[gt]  # remap HR labels
#
#        valid_mask = (gt != 0)
#        y_true_all.append(gt[valid_mask].flatten())
#        y_pred_all.append(output_hard[valid_mask].flatten())
#
#        with rasterio.open(lr_fn) as src:
#            lr = src.read(1)
#            lr = utils.LABEL_CLASS_TO_IDX_MAP[lr]  # remap LR labels
#
#        # === HR RGB for context ===
#        rgb_files = [f for f in os.listdir(args.hr_image_dir) if f.startswith(base_name)]
#        hr_rgb = None
#        if rgb_files:
#            with rasterio.open(os.path.join(args.hr_image_dir, rgb_files[0])) as src:
#                hr_rgb = np.moveaxis(src.read([1, 2, 3]), 0, -1)
#
#        pred_rgb = utils.class_map_to_rgb(output_hard)
#        gt_rgb   = utils.class_map_to_rgb(gt)
#        lr_rgb   = utils.class_map_to_rgb(lr)
#
#        # === Visualization with legend ===
#        fig, axs = plt.subplots(1, 4, figsize=(24, 6))
#        axs[0].imshow(hr_rgb) if hr_rgb is not None else axs[0].imshow(np.zeros_like(gt_rgb))
#        axs[0].set_title("HR Image")
#        axs[1].imshow(lr_rgb); axs[1].set_title("Low-Res Label")
#        axs[2].imshow(pred_rgb); axs[2].set_title("Prediction")
#        axs[3].imshow(gt_rgb); axs[3].set_title("Ground Truth")
#
#        for ax in axs:
#            ax.axis("off")
#
#        # Add legend
#        patches = [
#            mpatches.Patch(color=np.array(color)/255.0, label=utils.LABEL_NAMES[idx])
#            for idx, color in utils.LABEL_IDX_COLORMAP.items()
#        ]
#        fig.legend(handles=patches, loc="lower center", ncol=len(patches))
#
#        vis_fn = os.path.join(args.comparisons_dir, f"{base_name}_comparison.png")
#        plt.savefig(vis_fn, dpi=300, bbox_inches="tight")
#        plt.close()
#        print(f"Saved visualization: {vis_fn}")
#
#    # === Final metrics ===
#    y_true_all = np.concatenate(y_true_all)
#    y_pred_all = np.concatenate(y_pred_all)
#
#    oa = accuracy_score(y_true_all, y_pred_all)
#    kappa = cohen_kappa_score(y_true_all, y_pred_all)
#    miou = jaccard_score(y_true_all, y_pred_all, average="macro")
#    cm = confusion_matrix(y_true_all, y_pred_all, labels=list(range(1, args.num_classes)))
#
#    target_names = [utils.LABEL_NAMES[i] for i in range(1, args.num_classes)]
#    report = classification_report(
#        y_true_all, y_pred_all,
#        labels=list(range(1, args.num_classes)),
#        target_names=target_names,
#        digits=4
#    )
#
#    print("\n=== Final Metrics ===")
#    print(f"OA: {oa:.4f}, Kappa: {kappa:.4f}, mIoU: {miou:.4f}")
#    print("Confusion Matrix:\n", cm)
#    print(report)
#
#    with open(os.path.join(args.metrics_log, "metrics.txt"), "w") as f:
#        f.write(f"OA: {oa:.4f}\nKappa: {kappa:.4f}\nmIoU: {miou:.4f}\n")
#        f.write("Confusion Matrix:\n")
#        f.write(np.array2string(cm, separator=', '))
#        f.write("\n\nClassification Report:\n")
#        f.write(report)
#
#
## ==== Main ====
#if __name__ == "__main__":
#    parser = argparse.ArgumentParser()
#    parser.add_argument("--dataset", type=str, default="Chesapeake")
#    parser.add_argument("--list_dir", type=str, default=dataset_config["Chesapeake"]["list_dir"])
#    parser.add_argument("--num_classes", type=int, default=dataset_config["Chesapeake"]["num_classes"])
#    parser.add_argument("--model_path", type=str, default="/home/absingh/downloads/quantum/log_l2_qunet/v1-NYCtrain/unified/epoch_30.pth")
#    parser.add_argument("--save_path", type=str, default="/home/absingh/downloads/quantum/log_l2_qunet/v1-NYCtrain/unified/test/")
#    parser.add_argument("--metrics_log", type=str, default="/home/absingh/downloads/quantum/log_l2_qunet/v1-NYCtrain/unified/test/")
#    parser.add_argument("--comparisons_dir", type=str, default="/home/absingh/downloads/quantum/log_l2_qunet/v1-NYCtrain/unified/test/comparisons/")
#    parser.add_argument("--hr_label_dir", type=str, default="/home/absingh/downloads/quantum/dataset/Chesapeake_NewYork_dataset/test/HR_label")
#    parser.add_argument("--lr_label_dir", type=str, default="/home/absingh/downloads/quantum/dataset/Chesapeake_NewYork_dataset/test/LR_label")
#    parser.add_argument("--hr_image_dir", type=str, default="/home/absingh/downloads/quantum/dataset/Chesapeake_NewYork_dataset/test/HR_image")
#    parser.add_argument("--chip_size", type=int, default=128)
#    parser.add_argument("--chip_stride", type=int, default=64)
#    parser.add_argument("--gpu", type=str, default="0")
#    args = parser.parse_args()
#
#    if torch.cuda.is_available():
#        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
#        device = torch.device("cuda")
#        print(f"Using CUDA device(s): {args.gpu}")
#    else:
#        device = torch.device("cpu")
#        print("Using CPU")
#
#    model = QuantumUNet(num_classes=args.num_classes, in_channels=4).to(device)
#    model.load_state_dict(torch.load(args.model_path, map_location=device))
#
#    inference_and_eval(args, model)
