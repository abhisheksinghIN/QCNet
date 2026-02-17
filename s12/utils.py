import numpy as np
from matplotlib import colors

# ============================================================
#                  IMAGE NORMALIZATION (15 channels)
# ============================================================

IMAGE_MEANS = np.array([
    -12.27225428,
    -19.43526044,
    1234.8107257,
    997.61939167,
    903.69102444,
    754.84894802,
    969.18097088,
    1627.91563458,
    1913.47085481,
    1873.95219743,
    2086.10866809,
    710.37421625,
    11.20448001,
    1419.03241856,
    879.26727913
], dtype=np.float32)

IMAGE_STDS = np.array([
    5.18289882,
    6.61391834,
    194.32151121,
    260.6317736,
    300.12094349,
    456.76902268,
    475.76986072,
    876.19360379,
    1101.46597749,
    1122.17195714,
    1230.42499543,
    510.8142223,
    6.90812234,
    948.92739869,
    720.08974123
], dtype=np.float32)

IMAGE_MEANS_val = np.array([-13.94486718, -21.53669325, 1353.91294963, 1094.39913718, 999.56866115, 810.85222448, 1072.1886242, 1894.02925209, 2211.4349046, 2111.76325374, 2370.20395263, 637.68200383, 10.84459731, 1611.05854968, 962.27434458], dtype=np.float32)

IMAGE_STDS_val = np.array([4.32957283, 6.00005694, 170.30061843, 246.08313851, 299.51097874, 476.89079876, 499.13870013, 1056.70568738, 1336.41016601, 1299.67298585, 1490.16894983, 397.66470572, 4.59756429, 1095.24765291, 803.77610904], dtype=np.float32)
# Channel ordering for documentation and debugging
CHANNEL_ORDER = [
    "S1_VV", "S1_VH",
    "B2", "B3", "B4",
    "B8", "B11", "B12",
    "B1", "B5", "B6",
    "B7", "B8A",
    "B9", "B10"
]

# ============================================================
#     IGBP - DFC10 MAPPING (APPLIED TO ALL LABEL SOURCES)
# ============================================================

# DFC10 labels:
# 0=Ignore, 1=Forest, 2=Shrubland, 3=Savanna, 4=Grassland,
# 5=Wetlands, 6=Croplands, 7=Urban, 8=Snow/Ice, 9=Barren, 10=Water

ORIGINAL_TO_REMAPPED = {
    0: 0,        # unused
    1: 1, 2: 1, 3: 1, 4: 1, 5: 1,     # Forest
    6: 2, 7: 2,                       # Shrubland
    8: 3, 9: 3,                       # Savanna (optional masking)
    10: 4,                             # Grassland
    11: 5,                             # Wetlands
    12: 6, 14: 6,                      # Croplands
    13: 7,                             # Urban
    15: 8,                             # Snow/Ice
    16: 9,                             # Barren
    17: 10                             # Water
}


def get_label_class_to_idx_map():
    """Create a 0–255 lookup table for fast label remapping."""
    lut = np.zeros(256, dtype=np.int64)
    for k, v in ORIGINAL_TO_REMAPPED.items():
        lut[k] = v
    return lut


LABEL_CLASS_TO_IDX_MAP = get_label_class_to_idx_map()

# ==================== Class Mapping (HR GT) ==================== #
ORIGINAL_TO_REMAPPED_GT = {
    0: 0,          # Ignore
    1: 1,        # Forest
    2: 2,     # Shrubland
    3: 3,     # Savanna
    4: 4,      # Grassland
    5: 5,     # Wetlands
    6: 6,      # Croplands
    7: 7,    # Urban
    8: 8,    # Snow/Ice
    9: 9,      # Barren
    10: 10,   # Water
}

def get_label_class_to_idx_map_GT():
    label_to_idx_map = np.zeros(256, dtype=np.int64)
    for raw_label, new_class in ORIGINAL_TO_REMAPPED_GT.items():
        label_to_idx_map[raw_label] = new_class
    return label_to_idx_map

LABEL_CLASS_TO_IDX_MAP_GT = get_label_class_to_idx_map_GT()


# ============================================================
#                    COLORMAP (DFC10)
# ============================================================

LABEL_CLASS_COLORMAP = {
    0: (0, 0, 0),          # Ignore
    1: (0, 153, 0),        # Forest
    2: (198, 176, 68),     # Shrubland
    3: (251, 255, 19),     # Savanna
    4: (182, 255, 5),      # Grassland
    5: (39, 255, 135),     # Wetlands
    6: (194, 79, 68),      # Croplands
    7: (165, 165, 165),    # Urban
    8: (249, 255, 164),    # Snow/Ice
    #9: (28, 13, 255),      # Barren
    9: (150, 75, 0),        # Barren
    10: (0, 0, 255),   # Water
}

LABEL_IDX_COLORMAP = {k: v for k, v in LABEL_CLASS_COLORMAP.items()}

# ============================================================
#                    LABEL NAMES
# ============================================================

LABEL_NAMES = {
    0: "Ignore",
    1: "Forest",
    2: "Shrubland",
    3: "Savanna",
    4: "Grassland",
    5: "Wetlands",
    6: "Croplands",
    7: "Urban",
    8: "Snow/Ice",
    9: "Barren",
    10: "Water",
}

# ============================================================
#          RGB VISUALIZATION HELPERS
# ============================================================

def class_map_to_rgb(class_map):
    """Convert (H, W) class map to an RGB visualization."""
    rgb = np.zeros((*class_map.shape, 3), dtype=np.uint8)
    for cls_id, color in LABEL_CLASS_COLORMAP.items():
        rgb[class_map == cls_id] = color
    return rgb


# Optional helper (used by training script if needed)
#def mask_savanna(label_map):
#    """Mask savanna classes by setting them to ignore (0)."""
#    out = label_map.copy()
#    out[(out == 3)] = 0
#    return out
def mask_savanna(label_map):
    """
    Mask unwanted classes by setting them to ignore (0).
    Ignored classes: 2 (Shrubland), 3 (Savanna), 5 (Wetlands),
                     8 (Snow/Ice), 9 (Barren)
    """
    ignore_classes = {2, 3, 5, 8, 9}
    out = label_map.copy()
    for cls in ignore_classes:
        out[out == cls] = 0
    return out

