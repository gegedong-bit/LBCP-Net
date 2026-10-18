import torch
import torch.nn as nn
import torch.nn.functional as F


class HPGModule(nn.Module):
    def __init__(self, scales=3, tau=0.1):
        """
        Hierarchical Prior Generation module
        :param scales: The number of scales L for extracting the prior mask (default is 3)
        param tau: Temperature parameter of Softmax (default is 0.1)
        """
        super(HPGModule, self).__init__()
        self.scales = scales
        self.tau = tau

    def min_max_norm(self, x):
        """
        Perform Min-Max normalization on the feature map (F_norm in Eq.3)
        Ensure that the value range of each batch sample is between [0, 1]
        """
        B = x.size(0)
        x_flat = x.view(B, -1)
        mins = x_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        maxs = x_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        # 加上 1e-8 防止除以 0
        norm_x = (x - mins) / (maxs - mins + 1e-8)
        return norm_x

    def forward(self, F_h_s, F_h_q, M_s):
        """
        :param F_h_s: Support High-level feature [B, C, H, W]
        :param F_h_q: Query High-level feature [B, C, H, W]
        :param M_s: Support mask [B, 1, H0, W0]
        :return: multi_scale_prior_masks
        """
        prior_masks = []

        # Downsample the input Support Mask to the same spatial size as the high-level feature F_h_s
        B, C, H, W = F_h_s.shape
        #M_s_resized = F.interpolate(M_s, size=(H, W), mode='nearest')
        M_s_resized = F.interpolate(M_s, size=(H, W), mode='area')

        curr_F_s = F_h_s
        curr_F_q = F_h_q

        # Use the downsampled mask
        curr_M_s = M_s_resized

        for i in range(self.scales):
            B, C, H, W = curr_F_s.shape

            # curr_F_q_flat: [B, C, H*W] -> [B, H*W, C]
            curr_F_q_flat = curr_F_q.view(B, C, -1).transpose(1, 2)
            # curr_F_s_flat: [B, C, H*W]
            curr_F_s_flat = curr_F_s.view(B, C, -1)

            #  L2 Normalization
            norm_q = F.normalize(curr_F_q_flat, p=2, dim=-1)
            norm_s = F.normalize(curr_F_s_flat, p=2, dim=1)
            # A_q: [B, H*W, H*W]
            A_q = torch.bmm(norm_q, norm_s)

            #  tau = 0.1
            O_q = F.softmax(A_q / self.tau, dim=-1)

            #  support mask M_s -> [B, H*W, 1]
            M_s_flat = curr_M_s.view(B, -1, 1)


            #  [B, H*W, H*W] x [B, H*W, 1] -> [B, H*W, 1]
            prior_flat = torch.bmm(O_q, M_s_flat)
            # Reshape Min-Max
            prior_mask = prior_flat.view(B, 1, H, W)
            prior_mask = self.min_max_norm(prior_mask)

            prior_masks.append(prior_mask)

            if i < self.scales - 1:
                # Use average pooling to establish cross-scale information links
                weighted_F_q = curr_F_q * prior_mask
                # Synchronize and pool the feature map and Mask to achieve true weighted averaging
                sum_F_q = F.avg_pool2d(weighted_F_q, kernel_size=2, stride=2)
                sum_mask = F.avg_pool2d(prior_mask, kernel_size=2, stride=2) + 1e-8
                next_F_q = sum_F_q / sum_mask

                # Obtain the spatial size of the next scale
                next_size = next_F_q.shape[2:]

                next_F_s = F.interpolate(curr_F_s, size=next_size, mode='bilinear', align_corners=False)
                #next_M_s = F.interpolate(M_s, size=next_size, mode='nearest')
                next_M_s = F.interpolate(M_s, size=next_size, mode='area')

                curr_F_q = next_F_q
                curr_F_s = next_F_s
                curr_M_s = next_M_s

        return prior_masks
