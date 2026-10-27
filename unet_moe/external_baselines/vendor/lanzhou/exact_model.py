import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super().__init__()
        mid = max(1, in_channels // reduction_ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y_avg = self.avg_pool(x).view(b, c)
        y_max = self.max_pool(x).view(b, c)
        y_avg = self.fc(y_avg).view(b, c, 1, 1)
        y_max = self.fc(y_max).view(b, c, 1, 1)
        y = self.sigmoid(y_avg + y_max)
        return x * y


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg_out, max_out], dim=1)
        attn = self.conv(attn)
        attn = self.sigmoid(attn)
        return x * attn


class MultiScaleDenseBlock(nn.Module):
    def __init__(self, in_channels, reduction=4, norm_layer=None):
        super().__init__()
        mid_channels = in_channels // reduction
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.scale1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1),
            norm_layer(mid_channels),
            nn.LeakyReLU(0.01, inplace=True)
        )

        self.scale2 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            norm_layer(mid_channels),
            nn.LeakyReLU(0.01, inplace=True)
        )

        self.scale3 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=2, dilation=2),
            norm_layer(mid_channels),
            nn.LeakyReLU(0.01, inplace=True)
        )

        self.scale4 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=3, dilation=3),
            norm_layer(mid_channels),
            nn.LeakyReLU(0.01, inplace=True)
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(mid_channels * 4, in_channels, kernel_size=1),
            norm_layer(in_channels),
            nn.LeakyReLU(0.01, inplace=True)
        )

        self.ca = ChannelAttention(in_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        out1 = self.scale1(x)
        out2 = self.scale2(x)
        out3 = self.scale3(x)
        out4 = self.scale4(x)
        concat = torch.cat([out1, out2, out3, out4], dim=1)
        fused = self.fusion(concat)
        fused = self.ca(fused)
        fused = self.sa(fused)
        return x + fused


class ExactFusedSwinUNet(nn.Module):
    def __init__(self, seg_classes=2, classification_task_info=None, use_gn=False, gn_groups=8):
        super().__init__()
        def _get_norm_layer():
            if use_gn:
                def gn_layer(channels):
                    max_groups = min(gn_groups, channels)
                    for g in range(max_groups, 0, -1):
                        if channels % g == 0:
                            return nn.GroupNorm(g, channels)
                    return nn.GroupNorm(1, channels)
                return gn_layer
            else:
                return lambda channels: nn.BatchNorm2d(channels)

        self._norm_layer = _get_norm_layer()
        norm_layer = self._norm_layer

        swin_model = models.swin_t(weights=None)
        self.swin_dims = [96, 192, 384, 768]

        return_nodes = {
            'features.1': 'stage1',
            'features.3': 'stage2',
            'features.5': 'stage3',
            'features.7': 'stage4',
        }
        self.backbone = create_feature_extractor(swin_model, return_nodes=return_nodes)

        if classification_task_info:
            cls_in_dim = self.swin_dims[3]
            self.classification_heads = nn.ModuleDict()
            for task_name, num_classes in classification_task_info.items():
                self.classification_heads[task_name] = nn.Sequential(
                    nn.Linear(cls_in_dim, 512),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(512, num_classes)
                )

        self.deep_blocks = nn.ModuleList([
            MultiScaleDenseBlock(dim, norm_layer=norm_layer) for dim in self.swin_dims
        ])

        self.feature_enhancer = nn.Sequential(
            nn.Conv2d(self.swin_dims[3], self.swin_dims[3] // 2, kernel_size=3, padding=1),
            norm_layer(self.swin_dims[3] // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.swin_dims[3] // 2, self.swin_dims[3], kernel_size=3, padding=1),
            norm_layer(self.swin_dims[3]),
            nn.ReLU(inplace=True)
        )

        self.up1 = self._up_block(self.swin_dims[3], self.swin_dims[2], norm_layer)
        self.dec1 = self._conv_block(self.swin_dims[2] * 2, self.swin_dims[2], norm_layer)

        self.up2 = self._up_block(self.swin_dims[2], self.swin_dims[1], norm_layer)
        self.dec2 = self._conv_block(self.swin_dims[1] * 2, self.swin_dims[1], norm_layer)

        self.up3 = self._up_block(self.swin_dims[1], self.swin_dims[0], norm_layer)
        self.dec3 = self._conv_block(self.swin_dims[0] * 2, self.swin_dims[0], norm_layer)

        self.up4 = self._up_block(self.swin_dims[0], self.swin_dims[0] // 2, norm_layer)

        self.final_conv = nn.Conv2d(self.swin_dims[0] // 2, seg_classes, kernel_size=1)

        self.ds_conv1 = nn.Conv2d(self.swin_dims[2], seg_classes, kernel_size=1)
        self.ds_conv2 = nn.Conv2d(self.swin_dims[1], seg_classes, kernel_size=1)
        self.ds_conv3 = nn.Conv2d(self.swin_dims[0], seg_classes, kernel_size=1)

    def _conv_block(self, in_ch, out_ch, norm_layer):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            norm_layer(out_ch),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            norm_layer(out_ch),
            nn.LeakyReLU(0.01, inplace=True),
            ChannelAttention(out_ch),
            SpatialAttention()
        )

    def _up_block(self, in_ch, out_ch, norm_layer):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            norm_layer(out_ch),
            nn.LeakyReLU(0.01, inplace=True)
        )

    def _normalize_feat(self, tensor, expected_c):
        if tensor.dim() != 4:
            raise ValueError('Feature must be 4D tensor')
        if tensor.shape[1] == expected_c:
            return tensor
        if tensor.shape[-1] == expected_c:
            return tensor.permute(0, 3, 1, 2).contiguous()
        return tensor.permute(0, 3, 1, 2).contiguous()

    def forward(self, x, classification_task_name=None):
        initial_size = x.shape[-2:]
        features = self.backbone(x)

        s1 = self._normalize_feat(features['stage1'], self.swin_dims[0])
        s2 = self._normalize_feat(features['stage2'], self.swin_dims[1])
        s3 = self._normalize_feat(features['stage3'], self.swin_dims[2])
        s4 = self._normalize_feat(features['stage4'], self.swin_dims[3])

        s1 = self.deep_blocks[0](s1)
        s2 = self.deep_blocks[1](s2)
        s3 = self.deep_blocks[2](s3)
        s4 = self.deep_blocks[3](s4)
        s4 = self.feature_enhancer(s4)

        class_output = None
        if hasattr(self, 'classification_heads') and classification_task_name:
            p4 = F.adaptive_avg_pool2d(s4, 1).flatten(1)
            class_output = self.classification_heads[classification_task_name](p4)

        d1_up = self.up1(s4)
        s3_adj = F.interpolate(s3, size=d1_up.shape[2:], mode='bilinear', align_corners=True) if s3.shape[2:] != d1_up.shape[2:] else s3
        d1 = torch.cat([d1_up, s3_adj], dim=1)
        d1 = self.dec1(d1)
        ds1 = self.ds_conv1(d1)

        d2_up = self.up2(d1)
        s2_adj = F.interpolate(s2, size=d2_up.shape[2:], mode='bilinear', align_corners=True) if s2.shape[2:] != d2_up.shape[2:] else s2
        d2 = torch.cat([d2_up, s2_adj], dim=1)
        d2 = self.dec2(d2)
        ds2 = self.ds_conv2(d2)

        d3_up = self.up3(d2)
        s1_adj = F.interpolate(s1, size=d3_up.shape[2:], mode='bilinear', align_corners=True) if s1.shape[2:] != d3_up.shape[2:] else s1
        d3 = torch.cat([d3_up, s1_adj], dim=1)
        d3 = self.dec3(d3)
        ds3 = self.ds_conv3(d3)

        d4 = self.up4(d3)
        final_out = F.interpolate(d4, size=initial_size, mode='bilinear', align_corners=True)
        seg_output = self.final_conv(final_out)

        ds1 = F.interpolate(ds1, size=initial_size, mode='bilinear', align_corners=True)
        ds2 = F.interpolate(ds2, size=initial_size, mode='bilinear', align_corners=True)
        ds3 = F.interpolate(ds3, size=initial_size, mode='bilinear', align_corners=True)

        seg_output = seg_output + 0.4 * ds1 + 0.3 * ds2 + 0.3 * ds3

        return seg_output, class_output
