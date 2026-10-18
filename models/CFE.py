"""CFE with a 7x7 local background ring and no query gating.

The module keeps the contrast-guided residual branches, uses a dilated ring
around the support foreground as the primary background region, and updates
the query feature directly with the learned residual:

    F_q_tilde = F_q + delta_q
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CFEModule(nn.Module):
    """Foreground/background contrast-guided residual feature enhancement."""

    def __init__(self, in_channels=256, num_priors=3, bg_kernel_size=7):
        super().__init__()
        if bg_kernel_size < 3 or bg_kernel_size % 2 == 0:
            raise ValueError("bg_kernel_size must be an odd integer >= 3.")

        self.in_channels = in_channels
        self.num_priors = num_priors
        self.bg_kernel_size = bg_kernel_size
        self.eps = 1e-8

        # Support: Fs + Pfg + Pbg + Ms + contrast.
        self.delta_s = self._delta_branch(3 * in_channels + 2, in_channels)
        # Query: Fq + Pfg + Pbg + multi-scale priors + contrast.
        query_channels = 3 * in_channels + num_priors + 1
        self.delta_q = self._delta_branch(query_channels, in_channels)

        # Start as an identity mapping through zero residual outputs.
        for branch in (self.delta_s, self.delta_q):
            nn.init.zeros_(branch[-1].weight)
            nn.init.zeros_(branch[-1].bias)

    @staticmethod
    def _delta_branch(input_channels, output_channels):
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 1),
        )

    def _masked_average(self, features, mask):
        """Return a masked prototype [B,D,1,1] and a validity flag."""
        area = mask.float().sum(dim=(2, 3), keepdim=True)
        total = (features.float() * mask.float()).sum(dim=(2, 3), keepdim=True)
        valid = area > 0
        prototype = total / area.clamp_min(self.eps)
        prototype = torch.where(valid, prototype, torch.zeros_like(prototype))
        return prototype, valid

    def _local_background_mask(self, support_mask):
        """Create a foreground-excluding ring around the support foreground."""
        kernel = self.bg_kernel_size
        padding = kernel // 2
        support_fg = (support_mask > 0.5).to(support_mask.dtype)
        dilated = F.max_pool2d(
            support_fg,
            kernel_size=kernel,
            stride=1,
            padding=padding,
        )
        return (dilated - support_fg).clamp_(0.0, 1.0)

    def _contrast_map(self, features, foreground, background, valid):
        """Compute cos(F,Pfg) - cos(F,Pbg) at every feature location."""
        features = F.normalize(features.float(), p=2, dim=1, eps=self.eps)
        foreground = F.normalize(foreground, p=2, dim=1, eps=self.eps)
        background = F.normalize(background, p=2, dim=1, eps=self.eps)
        similarity_fg = (features * foreground).sum(dim=1, keepdim=True)
        similarity_bg = (features * background).sum(dim=1, keepdim=True)
        difference = similarity_fg - similarity_bg
        return torch.where(valid, difference, torch.zeros_like(difference))

    def forward(self, F_s, F_q, prior_masks, M_s):
        if F_s.ndim != 4 or F_q.ndim != 4:
            raise ValueError("Support and query features must have shape [B,D,H,W].")
        if F_s.shape[:2] != F_q.shape[:2] or F_s.size(1) != self.in_channels:
            raise ValueError("Support/query batch and channel sizes must match in_channels.")
        if M_s.ndim != 4 or M_s.shape[:2] != (F_s.size(0), 1):
            raise ValueError("Support mask must have shape [B,1,H0,W0].")
        if len(prior_masks) != self.num_priors:
            raise ValueError(f"Expected {self.num_priors} query prior masks.")

        support_mask = F.interpolate(
            M_s.to(device=F_s.device, dtype=torch.float32),
            size=F_s.shape[2:],
            mode="nearest",
        ).clamp(0.0, 1.0)

        foreground, valid_fg = self._masked_average(F_s, support_mask)

        # Use the local ring first. If it is empty, fall back to the global
        # non-foreground region so the background prototype remains valid.
        local_bg_mask = self._local_background_mask(support_mask)
        local_background, valid_local_bg = self._masked_average(
            F_s, local_bg_mask
        )
        global_background, valid_global_bg = self._masked_average(
            F_s, 1.0 - support_mask
        )
        background = torch.where(
            valid_local_bg,
            local_background,
            global_background,
        )
        valid_bg = valid_local_bg | valid_global_bg
        valid_pair = valid_fg & valid_bg
        contrast_s = self._contrast_map(F_s, foreground, background, valid_pair)
        contrast_q = self._contrast_map(F_q, foreground, background, valid_pair)

        resized_priors = []
        for prior in prior_masks:
            if prior.ndim != 4 or prior.shape[:2] != (F_q.size(0), 1):
                raise ValueError("Each query prior must have shape [B,1,H_i,W_i].")
            prior = prior.to(device=F_q.device, dtype=F_q.dtype)
            if prior.shape[2:] != F_q.shape[2:]:
                prior = F.interpolate(
                    prior,
                    size=F_q.shape[2:],
                    mode="bilinear",
                    align_corners=False,
                )
            resized_priors.append(prior)

        Z_s = torch.cat(
            [
                F_s,
                foreground.to(F_s.dtype).expand_as(F_s),
                background.to(F_s.dtype).expand_as(F_s),
                support_mask.to(F_s.dtype),
                contrast_s.to(F_s.dtype),
            ],
            dim=1,
        )
        Z_q = torch.cat(
            [
                F_q,
                foreground.to(F_q.dtype).expand_as(F_q),
                background.to(F_q.dtype).expand_as(F_q),
                *resized_priors,
                contrast_q.to(F_q.dtype),
            ],
            dim=1,
        )

        delta_s = self.delta_s(Z_s)
        delta_q = self.delta_q(Z_q)
        F_s_tilde = F_s + delta_s
        F_q_tilde = F_q + delta_q
        return F_s_tilde, F_q_tilde
