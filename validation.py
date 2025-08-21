import rasterio
import numpy as np
import os
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score, jaccard_score
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# === Configuration === #
predicted_dir = "/home/absingh/downloads/quantum/log_l2_qunet/v1-NYCtrain/6qbits-hybrid-PC/"
ground_truth_dir = "/home/absingh/downloads/quantum/dataset/Chesapeake_NewYork_dataset/test/HR_label"
lr_label_dir = "/home/absingh/downloads/quantum/dataset/Chesapeake_NewYork_dataset/test/LR_label"

hr_image_path = "/home/absingh/downloads/quantum/dataset/Chesapeake_NewYork_dataset/test/HR_image"

# === Save log to txt file === #
log_path = "/home/absingh/downloads/quantum/log_l2_qunet/v1-NYCtrain/6qbits-hybrid-PC/final_metrics_log-30epochs.txt"


remapped_output_dir = os.path.join(predicted_dir, "remap")
rgb_output_dir = os.path.join(predicted_dir, "rgb")


os.makedirs(remapped_output_dir, exist_ok=True)
os.makedirs(rgb_output_dir, exist_ok=True)

## === Class Mappings === #
#ORIGINAL_TO_REMAPPED_O = {
#    11: 1,  12: 1,  90: 1,  95: 1,
#    41: 2,  42: 2,  43: 2,
#    52: 3,  71: 3, 81: 3, 82: 3,
#    31: 4,
#    21: 5, 22: 5, 23: 5, 24: 5,
#}
#
#ORIGINAL_TO_REMAPPED = {
#    1: 1, 2: 1, 15: 1, 16: 1,
#    8: 2, 9: 2, 10: 2,
#    11: 3, 12: 3, 13: 3, 14: 3,
#    7: 4,
#    3: 5, 4: 5, 5: 5, 6: 5,
#}

# === Class Mappings === #
ORIGINAL_TO_REMAPPED_O = {
    11: 1,  #Open Water
    12: 1,  #Ice/Snow
    90: 2,  #Woody Wetlands
    95: 3,  #Herbaceous Wetlands
    41: 2,  #Deciduous Forest
    42: 2,  #Evergrean Forest
    43: 2,  #Mixed Forest
    52: 3,  #Shrub
    71: 3,  #Grassland/Herbaceous
    81: 3,  #Pasture
    82: 3,  #Cultivated Crops
    31: 3,  #Barren Land
    21: 5,  #Developed
    22: 5,  #Developed 
    23: 5,  #Developed 
    24: 5,  #Developed
}

ORIGINAL_TO_REMAPPED = {
    1: 1,  #Open Water 
    2: 1,  #Ice/Snow 
    15: 2, #Woody Wetlands 
    16: 3, #Herbaceous Wetlands
    8: 2,  #Deciduous Forest 
    9: 2,  #Evergrean Forest 
    10: 2, #Mixed Forest
    11: 3, #Shrub 
    12: 3, #Grassland/Herbaceous 
    13: 3, #Pasture 
    14: 3, #Cultivated Crops
    7: 3,  #Barren Land
    3: 5,  #Developed 
    4: 5,  #Developed 
    5: 5,  #Developed 
    6: 5,  #Developed
}

LABEL_CLASS_COLORMAP = {
    0:  (0, 0, 0),
    1: (70, 107, 159),
    2: (28, 95, 44),
    3: (104, 171, 95),
    4: (179, 172, 159),
    5: (235, 0, 0),
    6: (171, 0, 0)
}

def class_map_to_rgb(class_map, colormap):
    rgb = np.zeros((class_map.shape[0], class_map.shape[1], 3), dtype=np.uint8)
    for class_id, color in colormap.items():
        mask = class_map == class_id
        rgb[mask] = color
    return rgb

# Initialize lists OUTSIDE the loop to accumulate all predictions
y_true_all = []
y_pred_all = []

# === Loop Through Predicted Files === #
for filename in os.listdir(predicted_dir):
    if filename.endswith("_predictions-new.tif"):
        base_name = filename.replace("_predictions-new.tif", "")
        
        predicted_path = os.path.join(predicted_dir, filename)
        ground_truth_path = os.path.join(ground_truth_dir, f"{base_name}_lc.tif")
        lr_label_path = os.path.join(lr_label_dir, f"{base_name}_nlcd.tif")

        #print(f"\nProcessing: {base_name}")

        # Read predicted raster
        with rasterio.open(predicted_path) as src:
            predicted = src.read(1)
            meta = src.meta.copy()

        remapped = np.copy(predicted)
        for original, new_class in ORIGINAL_TO_REMAPPED.items():
            remapped[predicted == original] = new_class

        # Save remapped raster
        meta.update(dtype=rasterio.uint8, compress='lzw')
        remapped_output_path = os.path.join(remapped_output_dir, f"{base_name}_remap.tif")
        with rasterio.open(remapped_output_path, 'w', **meta) as dst:
            dst.write(remapped, 1)
        print("Reclassification complete. Saved at:", remapped_output_path)

        # Read ground truth
        with rasterio.open(ground_truth_path) as src:
            ground_truth = src.read(1)

        # Read and remap LR label
        with rasterio.open(lr_label_path) as src:
            lr_label = src.read(1)
        lr_remapped = np.copy(lr_label)
        for original, new_class in ORIGINAL_TO_REMAPPED_O.items():
            lr_remapped[lr_label == original] = new_class

        # Compute Metrics
        valid_mask = (ground_truth != 15) & (remapped != 15)
#******************************** Added here to compute the metrics on all the set of images **********************************************        
        y_true_all.append(ground_truth[valid_mask].flatten())
        y_pred_all.append(remapped[valid_mask].flatten())
##******************************************************************************************************************************************
        hr_image_files = [f for f in os.listdir(hr_image_path) if f.startswith(base_name)]
        hr_images = []
        
        for hr_file in hr_image_files:
            hr_image_file_path = os.path.join(hr_image_path, hr_file)
            with rasterio.open(hr_image_file_path) as src:
                hr_image = src.read([1, 2, 3])  # Read the first 3 bands
                hr_images.append(np.moveaxis(hr_image, 0, -1))  # Shape to (height, width, 3)
 
        # RGB Visualization for each HR image
        for idx, hr_rgb in enumerate(hr_images):
            pred_rgb = class_map_to_rgb(remapped, LABEL_CLASS_COLORMAP)
            gt_rgb = class_map_to_rgb(ground_truth, LABEL_CLASS_COLORMAP)
            lr_rgb = class_map_to_rgb(lr_remapped, LABEL_CLASS_COLORMAP)

            fig, axs = plt.subplots(1, 4, figsize=(24, 6))

            axs[0].imshow(hr_rgb)
            axs[0].set_title(f"HR Image")
            axs[0].axis("off")

            axs[1].imshow(lr_rgb)
            axs[1].set_title("Low-Resolution")
            axs[1].axis("off")        

            axs[2].imshow(pred_rgb)
            axs[2].set_title("Predicted")
            axs[2].axis("off")

            axs[3].imshow(gt_rgb)
            axs[3].set_title("Ground Truth")
            axs[3].axis("off")

            legend_elements = [
                Patch(facecolor=np.array(LABEL_CLASS_COLORMAP[1])/255, edgecolor='k', label='Water'),
                Patch(facecolor=np.array(LABEL_CLASS_COLORMAP[2])/255, edgecolor='k', label='Forest'),
                Patch(facecolor=np.array(LABEL_CLASS_COLORMAP[3])/255, edgecolor='k', label='Low Vegetation'),
                #Patch(facecolor=np.array(LABEL_CLASS_COLORMAP[4])/255, edgecolor='k', label='Barren'),
                Patch(facecolor=np.array(LABEL_CLASS_COLORMAP[5])/255, edgecolor='k', label='Impervious'),
            ]

            fig.legend(handles=legend_elements, loc='lower center', ncol=5, fontsize='large')
            plt.tight_layout(rect=[0, 0.05, 1, 1])

            rgb_output_path = os.path.join(rgb_output_dir, f"{base_name}_comparison_{idx + 1}.png")
            plt.savefig(rgb_output_path, dpi=300)
            plt.close()

            print(f"Saved RGB comparison at: {rgb_output_path}")            


# === Compute Metrics After All Images === #
y_true_all = np.concatenate(y_true_all)
y_pred_all = np.concatenate(y_pred_all)

oa = accuracy_score(y_true_all, y_pred_all)
cm = confusion_matrix(y_true_all, y_pred_all, labels=[1, 2, 3, 5])
kappa = cohen_kappa_score(y_true_all, y_pred_all)
miou = jaccard_score(y_true_all, y_pred_all, average='macro')
class_accuracies = cm.diagonal() / cm.sum(axis=1)
aa = np.mean(class_accuracies)
report = classification_report(y_true_all, y_pred_all, labels=[1, 2, 3, 5], target_names=["Water", "Forest", "Low-Vegetation/Crop-Fields", #"Barren", 
"Impervious"], digits=4)

class_accuracies = cm.diagonal() / cm.sum(axis=1)
class_names = ["Water", "Forest", "Low-Vegetation/Crop-Fields", "Impervious"]
for i, acc in enumerate(class_accuracies):
    print(f"Accuracy for {class_names[i]}: {acc:.4f}")
    
print("\n=== Final Metrics Across All Images ===")
print(f"OA: {oa:.4f}, AA: {aa:.4f}, Kappa: {kappa:.4f}, mIoU: {miou:.4f}")
print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n", report)


with open(log_path, "w") as f:
    f.write("=== Final Metrics Across All Images ===\n")
    f.write(f"OA: {oa:.4f}\n")
    f.write(f"AA: {aa:.4f}\n")
    f.write(f"Kappa: {kappa:.4f}\n")
    f.write(f"mIoU: {miou:.4f}\n")
    f.write(f"Confusion Matrix: {cm}\n")
    f.write(f"Classification Report: {report}\n")
    
    f.write("Class-wise Accuracies:\n")
    for i, acc in enumerate(class_accuracies):
        f.write(f"{class_names[i]}: {acc:.4f}\n")
    f.write(np.array2string(cm, separator=', '))
