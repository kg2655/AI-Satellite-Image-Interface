import torch
import torch.nn as nn
import torchvision.models as models

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class SiameseUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Use a pretrained ResNet18 as the encoder backbone
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # ------------------------
        # ENCODER (Siamese Branches)
        # ------------------------
        # Initial layers
        self.enc0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # 64 channels, 128x128
        self.maxpool = resnet.maxpool                                     # 64 channels, 64x64
        
        # ResNet Blocks
        self.enc1 = resnet.layer1  # 64 channels, 64x64
        self.enc2 = resnet.layer2  # 128 channels, 32x32
        self.enc3 = resnet.layer3  # 256 channels, 16x16
        self.enc4 = resnet.layer4  # 512 channels, 8x8
        
        # ------------------------
        # DECODER
        # ------------------------
        # Up 4
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(256 + 256, 256) # 256 from Up4 + 256 from Skip3
        
        # Up 3
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(128 + 128, 128) # 128 from Up3 + 128 from Skip2
        
        # Up 2
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(64 + 64, 64)    # 64 from Up2 + 64 from Skip1
        
        # Up 1
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(32 + 64, 32)    # 32 from Up1 + 64 from Skip0
        
        # Up 0 (Final upsample to original image size: 256x256)
        self.up0 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.conv0 = DoubleConv(16, 16)
        
        # Final prediction layer
        self.final = nn.Conv2d(16, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward_once(self, x):
        """Pass a single image through the encoder and collect skip connections"""
        e0 = self.enc0(x)
        e1 = self.enc1(self.maxpool(e0))
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        return e0, e1, e2, e3, e4

    def forward(self, x1, x2):
        # 1. Extract features independently (Siamese parallel processing)
        x1_e0, x1_e1, x1_e2, x1_e3, x1_e4 = self.forward_once(x1)
        x2_e0, x2_e1, x2_e2, x2_e3, x2_e4 = self.forward_once(x2)
        
        # 2. Calculate Absolute Difference at every skip level
        d0 = torch.abs(x1_e0 - x2_e0) # 128x128
        d1 = torch.abs(x1_e1 - x2_e1) # 64x64
        d2 = torch.abs(x1_e2 - x2_e2) # 32x32
        d3 = torch.abs(x1_e3 - x2_e3) # 16x16
        d4 = torch.abs(x1_e4 - x2_e4) # 8x8
        
        # 3. Decode & merge via Skip Connections
        u4 = self.up4(d4)
        c4 = self.conv4(torch.cat([u4, d3], dim=1)) # Skip Connection 3 merged
        
        u3 = self.up3(c4)
        c3 = self.conv3(torch.cat([u3, d2], dim=1)) # Skip Connection 2 merged
        
        u2 = self.up2(c3)
        c2 = self.conv2(torch.cat([u2, d1], dim=1)) # Skip Connection 1 merged
        
        u1 = self.up1(c2)
        c1 = self.conv1(torch.cat([u1, d0], dim=1)) # Skip Connection 0 merged
        
        u0 = self.up0(c1)
        c0 = self.conv0(u0) # Full 256x256 resolution
        
        # 4. Predict
        out = self.sigmoid(self.final(c0))
        return out
