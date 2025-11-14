import numpy as np

IMAGE_MEANS = np.array([117.67, 130.39, 121.52, 162.92])
IMAGE_STDS = np.array([39.25, 37.82, 24.24, 60.03])

# ==================== Class Mapping (LR) ==================== #
# Raw NLCD → Reduced classes
# Barren (31) → ignore (0) for training stability
ORIGINAL_TO_REMAPPED = {
    11: 1, 12: 1,               # water
    41: 2, 42: 2, 43: 2, 90: 2, # forest
    52: 3, 71: 3, 81: 3, 82: 3, 95: 3, # low veg
    31: 4,                      # barren → ignore ✅
    21: 5, 22: 5, 23: 5, 24: 5, # impervious
    #5: 5, 6: 5,                 # HR impervious merged
    15: 0                       # no-data
}

def get_label_class_to_idx_map():
    label_to_idx_map = np.zeros(256, dtype=np.int64)
    for raw_label, new_class in ORIGINAL_TO_REMAPPED.items():
        label_to_idx_map[raw_label] = new_class
    return label_to_idx_map

LABEL_CLASS_TO_IDX_MAP = get_label_class_to_idx_map()

# ==================== Class Mapping (HR GT) ==================== #
ORIGINAL_TO_REMAPPED_GT = {
    1: 1,  # water
    2: 2,  # forest
    3: 3,  # low veg
    4: 0,  # barren → ignore ✅
    5: 5,  # impervious
    6: 5,  # impervious
    15: 0  # no-data
}

def get_label_class_to_idx_map_GT():
    label_to_idx_map = np.zeros(256, dtype=np.int64)
    for raw_label, new_class in ORIGINAL_TO_REMAPPED_GT.items():
        label_to_idx_map[raw_label] = new_class
    return label_to_idx_map

LABEL_CLASS_TO_IDX_MAP_GT = get_label_class_to_idx_map_GT()

# ==================== Colormap ==================== #
LABEL_CLASS_COLORMAP = {
    0: (0, 0, 0),         # ignore/no-data (black)
    1: (70, 107, 159),    # water (blue)
    2: (28, 95, 44),      # forest (green)
    3: (144, 238, 144),   # low vegetation (light green)
    5: (235, 0, 0),       # impervious (red)
}

LABEL_IDX_COLORMAP = {idx: color for idx, color in LABEL_CLASS_COLORMAP.items()}

# ==================== Human-readable names ==================== #
LABEL_NAMES = {
    0: "Ignore",
    1: "Water",
    2: "Forest",
    3: "Low Veg.",
    5: "Impervious",
}

def class_map_to_rgb(class_map):
    """Convert class map (H, W) → RGB image (H, W, 3)."""
    rgb = np.zeros((*class_map.shape, 3), dtype=np.uint8)
    for cls_id, color in LABEL_CLASS_COLORMAP.items():
        rgb[class_map == cls_id] = color
    return rgb
