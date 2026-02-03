import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml


# -----------------------------
# Global config
# -----------------------------
GLOBAL_N_WIRES = 6  # Quantum circuit wires


# -----------------------------
# Quantum Convolutional Layer
# -----------------------------
class QuantumConv(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size=2, n_layers=4):
        super().__init__()
        self.n_wires = GLOBAL_N_WIRES
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        dev = qml.device("default.qubit", wires=self.n_wires)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def quantum_circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(self.n_wires))
            qml.StronglyEntanglingLayers(weights, wires=range(self.n_wires))
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_wires)]

        weight_shapes = {"weights": (n_layers, self.n_wires, 3)}
        self.q_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

        self.fc = nn.Linear(self.n_wires, out_channels)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        #B, C, H, W = x.shape
        B, C, H, W = x.shape
        p = self.patch_size
        # unfold patches
        x_unfold = x.unfold(2, p, p).unfold(3, p, p)
        x_unfold = x_unfold.contiguous().view(B, C, -1, p * p)
        x_mean = x_unfold.mean(dim=-1).permute(0, 2, 1).reshape(-1, C)

        # pad or truncate to n_wires
        if C < self.n_wires:
            pad = torch.zeros((x_mean.size(0), self.n_wires - C),
                              device=x.device, dtype=x_mean.dtype)
            x_in = torch.cat([x_mean, pad], dim=1)
        else:
            x_in = x_mean[:, :self.n_wires]

        # quantum layer
        q_out = torch.vmap(self.q_layer)(x_in)
        q_out = self.fc(q_out)
        q_out = self.norm(q_out)
        q_out = torch.relu(q_out)

        # reshape back to image
        q_out = q_out.view(B, H // p, W // p, self.out_channels).permute(0, 3, 1, 2)
        q_out = F.interpolate(q_out, scale_factor=p, mode='bilinear', align_corners=False)
        return q_out


# -----------------------------
# ConvBlock
# -----------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True):
        super().__init__()
        self.down = down
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        if down:
            self.pool = nn.MaxPool2d(2)
        else:
            self.up = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        if self.down:
            return x, self.pool(x)
        else:
            return self.up(x)


# -----------------------------
# Quantum U-Net Block
# -----------------------------
class QBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True):
        super().__init__()
        self.down = down
        self.q1 = QuantumConv(in_channels, out_channels)
        self.q2 = QuantumConv(out_channels, out_channels)
        if not down:
            self.resample = nn.ConvTranspose2d(out_channels, out_channels, 2, stride=2)
        else:
            self.resample = None

    def forward(self, x):
        x = self.q1(x)
        x = self.q2(x)
        if self.down:
            return x, F.avg_pool2d(x, 2, 2)
        else:
            return self.resample(x)


# -----------------------------
# Quantum U-Net
# -----------------------------
class QuantumUNet(nn.Module):
    def __init__(self, in_channels=4, num_classes=6):
        super().__init__()

        # ----- Encoder -----
        self.enc1a = ConvBlock(in_channels, 16, down=True)
        self.enc1b = ConvBlock(16, 16, down=True)   # stacked CNNs

        self.enc2 = QBlock(16, 32, down=True)
        self.enc3 = QBlock(32, 64, down=True)
        self.enc4 = QBlock(64, 128, down=True)

        self.bottleneck = nn.Sequential(
            QuantumConv(128, 256),
            QuantumConv(256, 256)
        )

        # ----- Decoder -----
        self.up3 = QBlock(256, 128, down=False)     # concat with enc4
        self.up2 = QBlock(256, 64, down=False)      # concat with enc3
        self.up1 = QBlock(128, 32, down=False)      # concat with enc2

        self.up0a = ConvBlock(64, 32, down=False)   # concat with enc1
        self.up0b = ConvBlock(48, 32, down=False)   # handles 32+16=48

        # ----- Heads -----
        self.final_conv = nn.Conv2d(32, num_classes, 1)
        self.recon_conv = nn.Conv2d(32, in_channels, 1)
        
        # NEW auxiliary classifier for pseudo labels
        self.aux_classifier = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1, x = self.enc1a(x)
        x1, x = self.enc1b(x)    # after 2nd convblock
        x2, x = self.enc2(x)
        x3, x = self.enc3(x)
        x4, x = self.enc4(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        # Decoder stage 3
        x = self.up3(x)
        if x.shape[2:] != x4.shape[2:]:
            x = F.interpolate(x, size=x4.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, x4], dim=1)
        
        # Decoder stage 2
        x = self.up2(x)
        if x.shape[2:] != x3.shape[2:]:
            x = F.interpolate(x, size=x3.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, x3], dim=1)
        
        # Decoder stage 1
        x = self.up1(x)
        if x.shape[2:] != x2.shape[2:]:
            x = F.interpolate(x, size=x2.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, x2], dim=1)
        
        # Final stage
        x = self.up0a(x)
        if x.shape[2:] != x1.shape[2:]:
            x = F.interpolate(x, size=x1.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, x1], dim=1)

        x = self.up0b(x)

        # Heads
        seg_logits = self.final_conv(x)
        recon = self.recon_conv(x)
        
        # auxiliary classifier for pseudo-labels
        aux_logits = self.aux_classifier(recon)  # [B, num_classes, H, W]
        
        return seg_logits, recon, aux_logits







#import torch
#import torch.nn as nn
#import pennylane as qml
#import torch.nn.functional as F
#
## -----------------------------
## Global config
## -----------------------------
#GLOBAL_N_WIRES = 6  # Quantum circuit wires
#
## -----------------------------
## Quantum Convolutional Layer
## -----------------------------
#class QuantumConv(nn.Module):
#    def __init__(self, in_channels, out_channels, patch_size=2, n_layers=4):
#        super().__init__()
#        self.in_channels = in_channels
#        self.out_channels = out_channels
#        self.patch_size = patch_size
#        self.n_wires = GLOBAL_N_WIRES
#        self.n_layers = n_layers
#
#        dev = qml.device("default.qubit", wires=self.n_wires)
#
#        @qml.qnode(dev, interface="torch", diff_method="backprop")
#        def quantum_circuit(inputs, weights):
#            qml.AngleEmbedding(inputs, wires=range(self.n_wires))
#            qml.StronglyEntanglingLayers(weights, wires=range(self.n_wires))
#            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_wires)]
#
#        weight_shapes = {"weights": (self.n_layers, self.n_wires, 3)}
#        self.q_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
#
#        self.fc = nn.Linear(self.n_wires, out_channels)
#        self.norm = nn.LayerNorm(out_channels)
#
#    def forward(self, x):
#        B, C, H, W = x.shape
#        p = self.patch_size
#        assert H % p == 0 and W % p == 0, "Image dimensions must be divisible by patch size"
#
#        x_unfold = x.unfold(2, p, p).unfold(3, p, p)
#        x_unfold = x_unfold.contiguous().view(B, C, -1, p * p)
#        x_mean = x_unfold.mean(dim=-1).permute(0, 2, 1).reshape(-1, C)
#
#        if C < self.n_wires:
#            pad = torch.zeros((x_mean.size(0), self.n_wires - C), device=x.device)
#            x_in = torch.cat([x_mean, pad], dim=1)
#        else:
#            x_in = x_mean[:, :self.n_wires]
#
#        q_out = torch.vmap(self.q_layer)(x_in)
#        q_out = self.fc(q_out)
#        q_out = self.norm(q_out)
#        q_out = torch.relu(q_out)
#
#        num_patches = (H // p) * (W // p)
#        q_out = q_out.view(B, H // p, W // p, self.out_channels).permute(0, 3, 1, 2)
#        return torch.nn.functional.interpolate(q_out, scale_factor=p, mode='bilinear', align_corners=False)
#
#class ConvBlock(nn.Module):
#    """Standard convolutional block (used for first and last U-Net blocks)."""
#    def __init__(self, in_channels, out_channels, down=True):
#        super().__init__()
#        self.down = down
#        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
#        self.bn1 = nn.BatchNorm2d(out_channels)
#        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
#        self.bn2 = nn.BatchNorm2d(out_channels)
#        self.relu = nn.ReLU(inplace=True)
#        if down:
#            self.pool = nn.MaxPool2d(2)
#        else:
#            self.up = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
#
#    def forward(self, x):
#        x = self.relu(self.bn1(self.conv1(x)))
#        x = self.relu(self.bn2(self.conv2(x)))
#        if self.down:
#            x_down = self.pool(x)
#            return x, x_down
#        else:
#            return self.up(x)
#
#
## -----------------------------
## Quantum U-Net Block
## -----------------------------
#class QBlock(nn.Module):
#    def __init__(self, in_channels, out_channels, down=True):
#        super().__init__()
#        self.down = down
#        self.q1 = QuantumConv(in_channels, out_channels)
#        self.q2 = QuantumConv(out_channels, out_channels)
#        self.resample = nn.Identity()
#
#        if not down:
#            self.resample = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
#
#    def forward(self, x):
#        x = self.q1(x)
#        x = self.q2(x)
#        if self.down:
#            return x, F.avg_pool2d(x, kernel_size=2, stride=2)
#        else:
#            return self.resample(x)
#
## -----------------------------
## Quantum U-Net
## -----------------------------
#class QuantumUNet(nn.Module):
#    def __init__(self, in_channels=1, num_classes=2):
#        super().__init__()
#
#        # Use standard conv for first and last
#        self.enc1 = ConvBlock(in_channels, 16, down=True)  # << classical
#        self.enc2 = QBlock(16, 32, down=True)
#        self.enc3 = QBlock(32, 64, down=True)
#        self.enc4 = QBlock(64, 128, down=True)
#
#        self.bottleneck = nn.Sequential(
#            QuantumConv(128, 256),
#            QuantumConv(256, 256)
#        )
#
#        self.up3 = QBlock(256, 128, down=False)
#        self.up2 = QBlock(128, 64, down=False)
#        self.up1 = QBlock(64, 32, down=False)
#        #self.up0 = ConvBlock(32, 16, down=False)  # << classical
#        self.up0 = ConvBlock(64, 16, down=False)  # <<< fix this
#
#        self.final_conv = nn.Conv2d(16 + 16, num_classes, kernel_size=1)
#
#    def forward(self, x):
#        x1, x = self.enc1(x)  # [B, 16, H/2, W/2]
#        x2, x = self.enc2(x)  # [B, 32, H/4, W/4]
#        x3, x = self.enc3(x)  # [B, 64, H/8, W/8]
#        x4, x = self.enc4(x)  # [B, 128, H/16, W/16]
#
#        x = self.bottleneck(x)  # [B, 256, H/16, W/16]
#
#        x = self.up3(x)         # [B, 128, H/8, W/8]
#        x = torch.cat([x, x4], dim=1)
#
#        x = self.up2(x)         # [B, 64, H/4, W/4]
#        x = torch.cat([x, x3], dim=1)
#
#        x = self.up1(x)         # [B, 32, H/2, W/2]
#        x = torch.cat([x, x2], dim=1)
#
#        x = self.up0(x)         # [B, 16, H, W]
#        x = torch.cat([x, x1], dim=1)
#
#
#
#        x = self.final_conv(x)  # [B, num_classes, H, W]
#        return x
#

