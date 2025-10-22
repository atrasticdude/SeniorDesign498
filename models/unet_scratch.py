# unet_from_scratch.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import random

# ---- Basic building blocks ----
class DoubleConv(nn.Module):
    """(conv -> BN -> ReLU) * 2"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=0),  # we'll crop/pack manually
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)

def center_crop(tensor, target_h, target_w):
    """Center-crop `tensor` (N,C,H,W) to (target_h, target_w)."""
    _, _, h, w = tensor.shape
    start_h = (h - target_h) // 2
    start_w = (w - target_w) // 2
    return tensor[:, :, start_h:start_h+target_h, start_w:start_w+target_w]

# ---- U-Net ----
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_features=64):
        super().__init__()
        f = base_features
        # Encoder
        self.enc1 = DoubleConv(in_channels, f)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(f, f*2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(f*2, f*4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DoubleConv(f*4, f*8)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(f*8, f*16)

        # Decoder (upsample -> concat -> double conv)
        self.up4 = nn.ConvTranspose2d(f*16, f*8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(f*16, f*8)
        self.up3 = nn.ConvTranspose2d(f*8, f*4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(f*8, f*4)
        self.up2 = nn.ConvTranspose2d(f*4, f*2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(f*4, f*2)
        self.up1 = nn.ConvTranspose2d(f*2, f, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(f*2, f)

        # Final conv (1x1)
        self.outconv = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder (note: DoubleConv reduces spatial size by 4 px per block because we used kernel_size=3 w/o padding)
        # To follow classic U-Net, input size must be such that cropping works (or we can pad beforehand).
        x1 = self.enc1(x)  # -> reduced
        p1 = self.pool1(x1)

        x2 = self.enc2(p1)
        p2 = self.pool2(x2)

        x3 = self.enc3(p2)
        p3 = self.pool3(x3)

        x4 = self.enc4(p3)
        p4 = self.pool4(x4)

        bott = self.bottleneck(p4)

        # Decoder + cropping of skip connections
        u4 = self.up4(bott)
        # crop x4 to u4's size
        x4_c = center_crop(x4, u4.shape[2], u4.shape[3])
        d4 = self.dec4(torch.cat([u4, x4_c], dim=1))

        u3 = self.up3(d4)
        x3_c = center_crop(x3, u3.shape[2], u3.shape[3])
        d3 = self.dec3(torch.cat([u3, x3_c], dim=1))

        u2 = self.up2(d3)
        x2_c = center_crop(x2, u2.shape[2], u2.shape[3])
        d2 = self.dec2(torch.cat([u2, x2_c], dim=1))

        u1 = self.up1(d2)
        x1_c = center_crop(x1, u1.shape[2], u1.shape[3])
        d1 = self.dec1(torch.cat([u1, x1_c], dim=1))

        out = self.outconv(d1)
        return out

# ---- Loss: BCE + Dice ----
def dice_coeff(pred, target, smooth=1e-6):
    # pred: logits or probabilities? we'll assume sigmoid applied outside as needed.
    pred = pred.contiguous().view(pred.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    intersection = (pred * target).sum(dim=1)
    dice = (2. * intersection + smooth) / (pred.sum(dim=1) + target.sum(dim=1) + smooth)
    return dice.mean()

class BCEDiceLoss(nn.Module):
    def __init__(self, weight_bce=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.w = weight_bce

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        dice = dice_coeff(probs, targets)
        return self.w * bce_loss + (1 - self.w) * (1 - dice)

# ---- Small example / quick test ----
def example_run():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Choose input size: because blocks reduce size by 4 px each DoubleConv,
    # ensure starting H,W are compatible. A safe choice: multiples of 16 plus margin.
    # Example: 572x572 was used in original U-Net; we'll pick 388x388 for smaller memory footprint.
    N = 4
    C_in = 1
    H = W = 388  # classic U-Net works with 572 -> cropping; pick 388 for memory
    model = UNet(in_channels=C_in, out_channels=1, base_features=32).to(device)

    # Synthetic dataset: random images and circles as masks (toy)
    x = torch.randn(N, C_in, H, W, device=device)
    # synthetic binary masks (random)
    y = (torch.rand(N, 1, H, W, device=device) > 0.5).float()

    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    criterion = BCEDiceLoss(weight_bce=0.5)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(2):  # tiny demo: 2 epochs
        total_loss = 0.0
        for xb, yb in loader:
            optim.zero_grad()
            logits = model(xb)
            # we must crop logits / targets to same spatial dims if mismatch occurs
            if logits.shape[2:] != yb.shape[2:]:
                # center crop target to logits size (or vice-versa) — choose crop target
                yb_c = center_crop(yb, logits.shape[2], logits.shape[3])
            else:
                yb_c = yb
            loss = criterion(logits, yb_c)
            loss.backward()
            optim.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, loss={total_loss/len(loader):.4f}")

    # single forward example and probabilities
    model.eval()
    with torch.no_grad():
        sample = x[:1]
        logits = model(sample)
        probs = torch.sigmoid(logits)
        print("Output shape:", probs.shape)  # (1,1,H',W')
    return model

if __name__ == "__main__":
    # run quick example if this script is run
    m = example_run()
