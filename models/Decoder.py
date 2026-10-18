import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Standard Residual Convolution Block"""

    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling"""

    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        dilations = [1, 6, 12, 18]
        self.aspp1 = nn.Conv2d(in_channels, out_channels, 1, padding=0, dilation=dilations[0], bias=False)
        self.aspp2 = nn.Conv2d(in_channels, out_channels, 3, padding=dilations[1], dilation=dilations[1], bias=False)
        self.aspp3 = nn.Conv2d(in_channels, out_channels, 3, padding=dilations[2], dilation=dilations[2], bias=False)
        self.aspp4 = nn.Conv2d(in_channels, out_channels, 3, padding=dilations[3], dilation=dilations[3], bias=False)
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=False)
        )
        self.conv_out = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x.size()[2:], mode='bilinear', align_corners=False)
        x_out = torch.cat((x1, x2, x3, x4, x5), dim=1)
        return self.conv_out(x_out)


class Decoder(nn.Module):
    def __init__(self, channels=256):
        super(Decoder, self).__init__()
        #  The number of input channels is 4D + 3 (3 Proxy(3D) + F_q_tilde(D) + 3 mask(3))
        in_channels = 4 * channels + 3

        #  3 residual blocks + ASPP
        self.res_blocks = nn.Sequential(
            ResidualBlock(in_channels, channels),
            ResidualBlock(channels, channels),
            ResidualBlock(channels, channels)
        )
        self.aspp = ASPP(channels, channels // 2)

        # Finally output the prediction mask (Logits) of 1 channel
        self.final_conv = nn.Conv2d(channels // 2, 1, kernel_size=1)

    def forward(self, proxies, F_q_tilde, prior_masks):
        B, C, H, W = F_q_tilde.shape

        # Expand and reshape the three proxies of [B, 1, C] into [B, C, H, W]
        expanded_proxies = [p.view(B, C, 1, 1).expand(-1, -1, H, W) for p in proxies]

        # Align the 3 prior_masks interpolations to H, W
        resized_masks = []
        for mask in prior_masks:
            if mask.shape[2:] != (H, W):
                mask = F.interpolate(mask, size=(H, W), mode='bilinear', align_corners=False)
            resized_masks.append(mask)

        # Splice along channel dimensions: [B, 4C+3, H, W]
        F_out_q = torch.cat(expanded_proxies + [F_q_tilde] + resized_masks, dim=1)

        # Decode prediction
        x = self.res_blocks(F_out_q)
        x = self.aspp(x)
        logits = self.final_conv(x)

        return logits