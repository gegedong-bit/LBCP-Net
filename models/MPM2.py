"""连续先验引导的 MPM：替代原版固定 0.5 硬阈值。

查询先验作为连续注意力权重，非空二值支持掩码保留原有语义。
全空掩码对应分支返回零贡献；其他分支、proxy 残差和 MLP 继续更新。
保留原参数名称和尺寸，兼容 MPM1 的 state_dict，但前向行为已改变。
HPG、支持掩码最近邻缩放、Decoder 和最终预测阈值不在本改动范围内。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedAttention(nn.Module):
    def __init__(self, channels):
        """
        Mask Attention Mechanism
        """
        super(MaskedAttention, self).__init__()

        self.phi = nn.Linear(channels, channels, bias=False)

    def forward(self, F_feat, P_proxy, mask):
        """
        :param F_feat:  [B, C, H, W]
        :param P_proxy:  [B, 1, C]
        :param mask: [B, 1, H, W]
        :return: [B, 1, C]
        """
        B, C, H, W = F_feat.shape
        if mask.shape != (B, 1, H, W):
            raise ValueError("mask must have shape [B, 1, H, W] matching features")
        if P_proxy.shape != (B, 1, C):
            raise ValueError("P_proxy must have shape [B, 1, C]")

        F_flat = F_feat.flatten(2).transpose(1, 2)
        F_proj = self.phi(F_flat)

        # Keep cosine normalization and weighted reduction stable under AMP.
        compute_dtype = torch.float64 if F_feat.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=F_feat.device.type, enabled=False):
            F_norm = F.normalize(F_flat.to(compute_dtype), p=2, dim=-1, eps=1e-6)
            P_norm = F.normalize(P_proxy.to(compute_dtype), p=2, dim=-1, eps=1e-6)
            similarity = torch.bmm(F_norm, P_norm.transpose(1, 2))

            # Continuous prior weighting: w_i = m_i exp(a_i) / sum_j m_j exp(a_j).
            # Binary support masks still strictly exclude background positions.
            confidence = mask.to(compute_dtype).flatten(2).transpose(1, 2).clamp(0, 1)
            affinity = torch.exp(similarity - similarity.amax(dim=1, keepdim=True))
            weighted_affinity = affinity * confidence
            mass = weighted_affinity.sum(dim=1, keepdim=True)
            valid = mass > 0

            # An empty mask contributes ZERO, rather than uniform global attention.
            # Avoid adding epsilon to the mask: doing so would admit background.
            denominator = torch.where(valid, mass, torch.ones_like(mass))
            attn_weights = torch.where(
                valid, weighted_affinity / denominator, torch.zeros_like(weighted_affinity))
            output = torch.bmm(attn_weights.transpose(1, 2), F_proj.to(compute_dtype))

        return output.to(F_proj.dtype)


class ProxyMiningBlock(nn.Module):
    def __init__(self, channels):
        """
        Proxy Mining Block (PMB)
        """
        super(ProxyMiningBlock, self).__init__()

        self.attn_s = MaskedAttention(channels)
        self.attn_q = MaskedAttention(channels)

        self.mlp = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels)
        )

    def forward(self, F_s, F_q, P_prev, M_s, M_q_prior):

        H_s = self.attn_s(F_s, P_prev, M_s)
        H_q = self.attn_q(F_q, P_prev, M_q_prior)

        # Integration and Update
        P_fused = P_prev + H_s + H_q
        P_new = P_fused + self.mlp(P_fused)

        return P_new


class MPMModule(nn.Module):
    def __init__(self, channels=256, scales=3):

        super(MPMModule, self).__init__()
        self.channels = channels
        self.scales = scales

        self.P0 = nn.Parameter(torch.randn(1, 1, channels))

        # PMB
        self.pmb_blocks = nn.ModuleList([ProxyMiningBlock(channels) for _ in range(scales)])

    def forward(self, F_s_tilde, F_q_tilde, prior_masks, M_s):

        B, C, H, W = F_s_tilde.shape

        multi_scale_F_s = [F_s_tilde]
        multi_scale_F_q = [F_q_tilde]
        multi_scale_M_s = [F.interpolate(M_s, size=(H, W), mode='nearest')]

        for i in range(1, self.scales):
            scale_factor = 2 ** i
            size = (H // scale_factor, W // scale_factor)

            multi_scale_F_s.append(F.adaptive_avg_pool2d(F_s_tilde, size))
            multi_scale_F_q.append(F.adaptive_avg_pool2d(F_q_tilde, size))
            multi_scale_M_s.append(F.interpolate(M_s, size=size, mode='nearest'))

        proxies = []

        curr_P = self.P0.expand(B, 1, self.channels)

        # Update Proxy
        for i in range(self.scales):
            target_size = multi_scale_F_q[i].shape[2:]
            curr_prior_mask = prior_masks[i]

            if curr_prior_mask.shape[2:] != target_size:
                curr_prior_mask = F.interpolate(
                    curr_prior_mask,
                    size=target_size,
                    mode='bilinear',
                    align_corners=False
                )
            # ========================================

            curr_P = self.pmb_blocks[i](
                multi_scale_F_s[i],
                multi_scale_F_q[i],
                curr_P,
                multi_scale_M_s[i],
                curr_prior_mask
            )
            proxies.append(curr_P)

        return proxies
