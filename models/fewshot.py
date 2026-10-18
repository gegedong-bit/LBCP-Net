import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from .CFE import CFEModule
from .HPG import HPGModule
from .MPM import MPMModule
from .Decoder import Decoder


class MPMNet(nn.Module):
    def __init__(self, pretrained=True, channels=256):
        super(MPMNet, self).__init__()
        # Backbone
        resnet = models.resnet101(pretrained=pretrained)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3

        self.mid_proj = nn.Sequential(nn.Conv2d(512, channels, 1, bias=False), nn.BatchNorm2d(channels),
                                      nn.ReLU(inplace=True))
        self.high_proj = nn.Sequential(nn.Conv2d(1024, channels, 1, bias=False), nn.BatchNorm2d(channels),
                                       nn.ReLU(inplace=True))


        self.hpg_module = HPGModule(scales=3, tau=0.1)
        self.cfe_module = CFEModule(in_channels=channels)
        self.mpm_module = MPMModule(channels=channels, scales=3)
        self.decoder = Decoder(channels=channels)

        self.criterion_bce = nn.BCEWithLogitsLoss()

    def extract_features(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)

        # 1. Get the original features of layer2 (B, 512, H/8, W/8)
        feat2 = self.layer2(x)

        # 2. Generate mid-level features: reduce feature dimension from 512 to 256 channels
        mid_feat = self.mid_proj(feat2)

        # 3. Get the original features of layer3: feed the output of layer2 (512 channels) to layer3
        # The output dimensions are (B, 1024, H/16, W/16)
        feat3 = self.layer3(feat2)

        # 4. Generate high-level features: reduce feature dimension from 1024 to 256 channels
        high_feat = self.high_proj(feat3)

        return mid_feat, high_feat

    def forward_single_direction(self, support_img, query_img, support_mask):
        """One-way reasoning process: using Support to guide Query segmentation"""
        # Feature extraction
        F_s, F_h_s = self.extract_features(support_img)
        F_q, F_h_q = self.extract_features(query_img)

        # Core module data flow
        prior_masks = self.hpg_module(F_h_s, F_h_q, support_mask)
        F_s_tilde, F_q_tilde = self.cfe_module(F_s, F_q, prior_masks, support_mask)
        proxies = self.mpm_module(F_s_tilde, F_q_tilde, prior_masks, support_mask)

        # Decode to get Logits
        logits = self.decoder(proxies, F_q_tilde, prior_masks)

        # Interpolate back to original image size [B, 1, 256, 256]
        logits = F.interpolate(logits, size=query_img.size()[2:], mode='bilinear', align_corners=False)
        return logits, proxies, F_s

    def forward(self, support_img, query_img, support_mask, query_mask=None, is_training=True):
        """
        In training mode, automatically perform role reversal calculations to compute PAR Loss.
        """
        # 1. forward propagation (Support -> Query)
        logits_q, proxies, F_s = self.forward_single_direction(support_img, query_img, support_mask)

        if not is_training or query_mask is None:
            return torch.sigmoid(logits_q)  # The inference stage directly returns the probability mask

        # ================= Training phase: loss calculation =================
        # L_bce
        loss_bce = self.criterion_bce(logits_q, query_mask)

        # Reverse the roles of Support and Query to mitigate intra-class differences
        logits_s, _, _ = self.forward_single_direction(query_img, support_img, query_mask)
        loss_par = self.criterion_bce(logits_s, support_mask)

        # Extract the foreground and background prototypes of Support
        B, C, H, W = F_s.shape
        M_s_resized = F.interpolate(support_mask, size=(H, W), mode='nearest')

        # Foreground prototype P_fg^s
        P_fg_s = (F_s * M_s_resized).sum(dim=(2, 3)) / (M_s_resized.sum(dim=(2, 3)) + 1e-8)  # [B, C]
        # Background prototype P_bg^s (reverse Mask)
        M_bg_resized = 1.0 - M_s_resized
        P_bg_s = (F_s * M_bg_resized).sum(dim=(2, 3)) / (M_bg_resized.sum(dim=(2, 3)) + 1e-8)  # [B, C]

        loss_pt = 0.0
        epsilon = 0.1  # Margin parameter of Triplet Loss (adjustable)
        for p in proxies:
            # The dimensions of p are [B, 1, C], reshaped to [B, C]
            p = p.view(B, C)
            # Calculate the Euclidean distance D
            dist_fg = F.pairwise_distance(p, P_fg_s, p=2)
            dist_bg = F.pairwise_distance(p, P_bg_s, p=2)
            # max(D(P, P_fg) - D(P, P_bg) + epsilon, 0)
            margin_loss = F.relu(dist_fg - dist_bg + epsilon)
            loss_pt += margin_loss.mean()

        #  \alpha_1 = 1.0, \alpha_2 = 1.0 [cite: 299]
        # alpha_1, alpha_2 = 1.0, 1.0
        # total_loss = loss_bce + alpha_1 * loss_par + alpha_2 * loss_pt

        return loss_bce, loss_par, loss_pt, torch.sigmoid(logits_q)


# Test the complete process
if __name__ == "__main__":
    B, C_img, H, W = 2, 3, 256, 256
    dummy_support = torch.randn(B, C_img, H, W)
    dummy_query = torch.randn(B, C_img, H, W)
    dummy_s_mask = torch.randint(0, 2, (B, 1, H, W)).float()
    dummy_q_mask = torch.randint(0, 2, (B, 1, H, W)).float()

    # Instantiate the ultimate model
    model = MPMNet(pretrained=False)

    # Simulation Training Steps
    loss, pred_mask = model(dummy_support, dummy_query, dummy_s_mask, dummy_q_mask, is_training=True)

    print(f"Total Combined Loss: {loss.item():.4f}")
    print(f"Final Predicted Mask Shape: {pred_mask.shape}")  #  [2, 1, 256, 256]
