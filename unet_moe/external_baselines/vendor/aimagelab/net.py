import torch
from torch import nn
from torchvision import models
from .utils import organ_to_class_dict, pad_to_2d, unpad_2d
import numpy as np


class FiLM2d(nn.Module):
    """
    Feature-wise Linear Modulation for a 2-D feature map.
    (γ, β) are generated from a learned embedding of organ_id.
    """

    def __init__(
        self,
        n_organs: int,
        in_channels: int,
        emb_dim: int = 64,
        hidden: int | None = None,
    ):
        super().__init__()
        hidden = hidden or 2 * in_channels
        self.embed = nn.Embedding(n_organs, emb_dim)

        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2 * in_channels),  # → [β‖γ]
        )

        # initialise so that FiLM starts as identity: γ≈1, β≈0
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.constant_(self.mlp[-1].bias[:in_channels], 0)  # β
        nn.init.constant_(self.mlp[-1].bias[in_channels:], 1)  # γ

    def forward(self, x: torch.Tensor, organ_id: torch.Tensor):
        """
        x : (B, C, H, W)
        organ_id : (B,) integer 0…n_organs-1
        """
        beta_gamma = self.mlp(self.embed(organ_id))  # (B, 2C)
        beta, gamma = beta_gamma.chunk(2, dim=-1)  # each (B, C)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


class OmniClsCBAM(nn.Module):
    def __init__(
        self,
        resnet_size="18",
        use_cbam=True,
        use_film=False,
        n_organs=len(organ_to_class_dict.keys()),
        num_classes=10,
        predict_bboxes=False,
        mlp_organ=False,
    ):
        """
        Simplified OmniClsCBAM with optional CBAM and FiLM modules.

        Args:
            resnet_size (str): ResNet architecture size ('18', '34', '50', '101', '152')
            use_cbam (bool): Whether to use CBAM attention modules
            use_film (bool): Whether to use FiLM conditioning layers
            n_organs (int): Number of different organs for FiLM conditioning
            num_classes (int): Number of output classes
        """
        super().__init__()

        self.use_cbam = use_cbam
        self.use_film = use_film
        self.predict_bboxes = predict_bboxes
        self.mlp_organ = mlp_organ

        resnet_map = {
            "18": models.resnet18,
            "34": models.resnet34,
            "50": models.resnet50,
            "101": models.resnet101,
            "152": models.resnet152,
        }
        if resnet_size not in resnet_map:
            raise ValueError(
                f"Unsupported resnet_size '{resnet_size}'. Choose from {list(resnet_map.keys())}."
            )

        backbone_model = resnet_map[resnet_size]()

        # Helper function to get the output channels of a layer
        def get_layer_channels(layer):
            last_block = layer[-1]
            if hasattr(last_block, "conv3"):  # Bottleneck
                return last_block.conv3.out_channels
            else:  # BasicBlock
                return last_block.conv2.out_channels

        # Build backbone components
        self.stem = nn.Sequential(
            backbone_model.conv1,
            backbone_model.bn1,
            backbone_model.relu,
            backbone_model.maxpool,
        )

        self.layer1 = backbone_model.layer1
        self.layer2 = backbone_model.layer2
        self.layer3 = backbone_model.layer3
        self.layer4 = backbone_model.layer4
        self.avgpool = backbone_model.avgpool

        # Add CBAM modules if requested
        if use_cbam:
            self.cbam1 = CBAM(get_layer_channels(backbone_model.layer1))
            self.cbam2 = CBAM(get_layer_channels(backbone_model.layer2))
            self.cbam3 = CBAM(get_layer_channels(backbone_model.layer3))
            self.cbam4 = CBAM(get_layer_channels(backbone_model.layer4))

        # Add FiLM modules if requested
        if use_film:
            self.film1 = FiLM2d(n_organs, get_layer_channels(backbone_model.layer1))
            self.film2 = FiLM2d(n_organs, get_layer_channels(backbone_model.layer2))
            self.film3 = FiLM2d(n_organs, get_layer_channels(backbone_model.layer3))
            self.film4 = FiLM2d(n_organs, get_layer_channels(backbone_model.layer4))

        self.embed_dim = backbone_model.fc.in_features + (64 if self.mlp_organ else 0)
        self.organ_mlp = nn.Sequential(
            nn.Embedding(n_organs, 64),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),  # Output dim to concat, e.g., 64
        )
        self.multi_cls = nn.Linear(self.embed_dim, num_classes)  # Adjust in_features
        # self.proj_head = nn.Linear(self.embed_dim, 1024)  # Adjust in_features
        self.bbox_regr = nn.Linear(self.embed_dim, 4)  # Adjust similarly
        self.dropout = nn.Dropout(p=0.2)

        self.initialize_weights_()

    def initialize_weights_(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def get_feat(self, x, organ_id=None):
        if self.use_film and organ_id is None:
            raise ValueError("organ_id must be provided when use_film=True")

        x = self.stem(x)

        # Layer 1
        x = self.layer1(x)
        if self.use_cbam:
            x = self.cbam1(x)
        if self.use_film:
            x = self.film1(x, organ_id)

        # Layer 2
        x = self.layer2(x)
        if self.use_cbam:
            x = self.cbam2(x)
        if self.use_film:
            x = self.film2(x, organ_id)

        # Layer 3
        x = self.layer3(x)
        if self.use_cbam:
            x = self.cbam3(x)
        if self.use_film:
            x = self.film3(x, organ_id)

        # Layer 4
        x = self.layer4(x)
        if self.use_cbam:
            x = self.cbam4(x)
        if self.use_film:
            x = self.film4(x, organ_id)

        # Final layers
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x

    def forward(self, x, organ_id=None):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input images (B, C, H, W)
            organ_id (torch.Tensor, optional): Organ IDs for FiLM conditioning (B,)

        Returns:
            tuple: (multi_cls_out, bbox_out)
        """
        if self.use_film and organ_id is None:
            raise ValueError("organ_id must be provided when use_film=True")

        x = self.stem(x)

        # Layer 1
        x = self.layer1(x)
        if self.use_cbam:
            x = self.cbam1(x)
        if self.use_film:
            x = self.film1(x, organ_id)

        # Layer 2
        x = self.layer2(x)
        if self.use_cbam:
            x = self.cbam2(x)
        if self.use_film:
            x = self.film2(x, organ_id)

        # Layer 3
        x = self.layer3(x)
        if self.use_cbam:
            x = self.cbam3(x)
        if self.use_film:
            x = self.film3(x, organ_id)

        # Layer 4
        x = self.layer4(x)
        if self.use_cbam:
            x = self.cbam4(x)
        if self.use_film:
            x = self.film4(x, organ_id)

        # Final layers
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # organ_emb = self.organ_embed(organ_id)  # (B, 64)
        if self.mlp_organ:
            organ_emb = self.organ_mlp(organ_id)  # (B, 64)
            x = torch.cat([x, organ_emb], dim=1)  # (B, embed_dim + 64)
        x = self.dropout(x)

        multi_cls_out = self.multi_cls(x)
        bbox_out = self.bbox_regr(x)
        # features_proj = self.proj_head(x)

        return multi_cls_out, bbox_out


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        conv_kwargs={"kernel_size": 3, "stride": 1, "padding": 1},
    ):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **conv_kwargs)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.activ = nn.LeakyReLU()
        self.block = nn.Sequential(self.conv, self.norm, self.activ)

    def forward(self, x):
        return self.block(x)


# Single encoder block
class DownConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: list,
        out_channels: list,
        conv_kwargs={"kernel_size": 3, "stride": 1, "padding": 1},
    ):
        super(DownConvBlock, self).__init__()

        assert len(in_channels) == len(
            out_channels
        ), f"in_channels length is {len(in_channels)} while out_channels is {len(out_channels)}"

        # Variable number of convolutional block in each layer, based on the in_channels and out_channels length
        self.conv_blocks = nn.ModuleList(
            [
                ConvBlock(in_ch, out_ch, conv_kwargs)
                for in_ch, out_ch in zip(in_channels, out_channels)
            ]
        )

        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        for block in self.conv_blocks:
            x = block(x)
        return self.pool(x), x


# Single decoder block
class UpConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: list,
        out_channels: list,
        up_conv=True,
        conv_kwargs={"kernel_size": 3, "stride": 1, "padding": 1},
        upconv_kwargs={"kernel_size": 2, "stride": 2},
    ):
        super(UpConvBlock, self).__init__()

        assert len(in_channels) == len(
            out_channels
        ), f"in_channels length is {len(in_channels)} while out_channels is {len(out_channels)}"

        # Variable number of convolutional block in each layer, based on the in_channels and out_channels length
        self.conv_blocks = nn.ModuleList(
            [
                ConvBlock(in_ch, out_ch, conv_kwargs)
                for in_ch, out_ch in zip(in_channels, out_channels)
            ]
        )

        self.up_conv = up_conv
        if self.up_conv:
            self.up_conv_op = nn.ConvTranspose2d(
                out_channels[-1], out_channels[-1], **upconv_kwargs
            )

    def forward(self, x):
        for block in self.conv_blocks:
            x = block(x)

        if self.up_conv:
            return self.up_conv_op(x)
        else:
            return x


class FiLM2d(nn.Module):
    """
    Feature-wise Linear Modulation for a 2-D feature map.
    (γ, β) are generated from a learned embedding of organ_id.
    """

    def __init__(
        self,
        n_organs: int,
        in_channels: int,
        emb_dim: int = 64,
        hidden: int | None = None,
    ):
        super().__init__()
        hidden = hidden or 2 * in_channels
        self.embed = nn.Embedding(n_organs, emb_dim)

        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2 * in_channels),  # → [β‖γ]
        )

        # initialise so that FiLM starts as identity: γ≈1, β≈0
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.constant_(self.mlp[-1].bias[:in_channels], 0)  # β
        nn.init.constant_(self.mlp[-1].bias[in_channels:], 1)  # γ

    def forward(self, x: torch.Tensor, organ_id: torch.Tensor):
        """
        x : (B, C, H, W)
        organ_id : (B,) integer 0…n_organs-1
        """
        beta_gamma = self.mlp(self.embed(organ_id))  # (B, 2C)
        beta, gamma = beta_gamma.chunk(2, dim=-1)  # each (B, C)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta


class DownConvBlockFiLM(nn.Module):
    """
    Conv → FiLM → Conv → FiLM → Pool.
    Except for FiLM, the API and behaviour remain the same.
    """

    def __init__(
        self,
        in_channels: list[int],
        out_channels: list[int],
        n_organs: int,
        conv_kwargs={"kernel_size": 3, "stride": 1, "padding": 1},
        emb_dim: int = 64,
    ):
        super().__init__()
        assert len(in_channels) == len(
            out_channels
        ), f"in_channels length is {len(in_channels)} while out_channels is {len(out_channels)}"

        self.conv_blocks = nn.ModuleList(
            [
                ConvBlock(in_ch, out_ch, conv_kwargs)
                for in_ch, out_ch in zip(in_channels, out_channels)
            ]
        )

        self.film_blocks = nn.ModuleList(
            [
                FiLM2d(n_organs=n_organs, in_channels=out_ch, emb_dim=emb_dim)
                for out_ch in out_channels
            ]
        )

        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x: torch.Tensor, organ_id: torch.Tensor):
        """
        organ_id : (B,) torch.long   (0 = breast, 1 = thyroid, …)
        """
        for conv, film in zip(self.conv_blocks, self.film_blocks):
            x = conv(x)  # usual conv-norm-ReLU
            x = film(x, organ_id)  # FiLM modulation
        return self.pool(x), x  # (downsampled, skip-connection)


class UpConvBlockFiLM(nn.Module):
    """
    Up-sampling block with FiLM conditioning.

    Conv → FiLM → Conv → FiLM → (optional) ConvTranspose2d
    """

    def __init__(
        self,
        in_channels: list[int],
        out_channels: list[int],
        n_organs: int,
        up_conv: bool = True,
        conv_kwargs: dict = {"kernel_size": 3, "stride": 1, "padding": 1},
        upconv_kwargs: dict = {"kernel_size": 2, "stride": 2},
        emb_dim: int = 64,
    ):
        super().__init__()
        assert len(in_channels) == len(
            out_channels
        ), f"in_channels length is {len(in_channels)} while out_channels is {len(out_channels)}"

        self.conv_blocks = nn.ModuleList(
            [
                ConvBlock(in_ch, out_ch, conv_kwargs)
                for in_ch, out_ch in zip(in_channels, out_channels)
            ]
        )

        self.film_blocks = nn.ModuleList(
            [
                FiLM2d(n_organs=n_organs, in_channels=out_ch, emb_dim=emb_dim)
                for out_ch in out_channels
            ]
        )

        self.up_conv = up_conv
        if self.up_conv:
            self.up_conv_op = nn.ConvTranspose2d(
                out_channels[-1], out_channels[-1], **upconv_kwargs
            )

    def forward(self, x: torch.Tensor, organ_id: torch.Tensor):
        """
        x : (B, C, H, W)
        organ_id : (B,) long tensor - 0=breast, 1=thyroid, …
        """
        for conv, film in zip(self.conv_blocks, self.film_blocks):
            x = conv(x)
            x = film(x, organ_id)

        if self.up_conv:
            x = self.up_conv_op(x)

        return x


class UNet2DFiLM(nn.Module):
    def __init__(self, in_channels, num_classes, n_organs, size=32, depth=3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = num_classes
        self.size = size
        self.depth = depth

        self.encoder = nn.ModuleDict()
        self.encoder["0"] = DownConvBlockFiLM(
            [self.in_channels, self.size], [self.size, self.size * 2], n_organs=n_organs
        )
        for i in range(1, self.depth):
            self.encoder[str(i)] = DownConvBlockFiLM(
                [self.size * (2**i), self.size * (2**i)],
                [self.size * (2**i), self.size * (2 ** (i + 1))],
                n_organs=n_organs,
            )

        self.bottleneck = UpConvBlockFiLM(
            [self.size * (2**self.depth), self.size * (2**self.depth)],
            [self.size * (2**self.depth), self.size * (2 ** (self.depth + 1))],
            n_organs=n_organs,
        )

        self.decoder = nn.ModuleDict()
        for i in range(self.depth, 1, -1):
            self.decoder[str(i - 1)] = UpConvBlockFiLM(
                [self.size * (2 ** (i + 1)) + self.size * (2**i), self.size * (2**i)],
                [self.size * (2**i), self.size * (2**i)],
                n_organs=n_organs,
            )
        self.decoder["0"] = UpConvBlockFiLM(
            [self.size * 4 + self.size * 2, self.size * 2],
            [self.size * 2, self.size * 2],
            n_organs=n_organs,
            up_conv=False,
        )
        self.out_layer = ConvBlock(
            self.size * 2,
            self.out_channels,
            conv_kwargs={"kernel_size": 1, "stride": 1, "padding": 0},
        )

    def encode(self, x, organ_id):
        feat_list = []

        # Padding is needed if the size of the input is not divisible 'depth' times by 2
        pre_padding = (
            (x.size(-1) % 2**self.depth != 0)
            or (x.size(-2) % 2**self.depth != 0)
        )
        pads = (0, 0, 0, 0)
        if pre_padding:
            x, pads = pad_to_2d(x, 2**self.depth)

        out, feat = self.encoder["0"](x, organ_id)
        feat_list.append(feat)

        for block in list(self.encoder)[1:]:
            out, feat = self.encoder[block](out, organ_id)
            feat_list.append(feat)

        out = self.bottleneck(out, organ_id)

        return out, feat_list, pads

    def decode(self, x, organ_id, out, feat_list, pads):
        pre_padding = (
            (x.size(-1) % 2**self.depth != 0)
            or (x.size(-2) % 2**self.depth != 0)
        )
        for block in self.decoder:
            out = self.decoder[block](
                torch.cat((out, feat_list[int(block)]), dim=1), organ_id
            )
            del feat_list[int(block)]

        out = self.out_layer(out)

        if pre_padding:
            out = unpad_2d(out, pads)

        return out

    def forward(self, x, organ_id):
        feat_list = []

        # Padding is needed if the size of the input is not divisible 'depth' times by 2
        pre_padding = (
            (x.size(-1) % 2**self.depth != 0)
            or (x.size(-2) % 2**self.depth != 0)
        )
        pads = (0, 0, 0, 0)
        if pre_padding:
            x, pads = pad_to_2d(x, 2**self.depth)

        out, feat = self.encoder["0"](x, organ_id)
        feat_list.append(feat)

        for block in list(self.encoder)[1:]:
            out, feat = self.encoder[block](out, organ_id)
            feat_list.append(feat)

        out = self.bottleneck(out, organ_id)

        for block in self.decoder:
            out = self.decoder[block](
                torch.cat((out, feat_list[int(block)]), dim=1), organ_id
            )
            del feat_list[int(block)]

        out = self.out_layer(out)

        if pre_padding:
            out = unpad_2d(out, pads)

        return out

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + "\nTrainable parameters: {}".format(params)


class FusionWrapper(nn.Module):
    def __init__(
        self,
        resnet_size,
        use_cbam,
        use_film,
        n_organs_cls,
        num_classes_cls,
        predict_bboxes,
        mlp_organ,
        in_channels,
        num_classes,
        n_organs,
        size=32,
        depth=3,
    ):
        super().__init__()

        self.cls_net = OmniClsCBAM(
            resnet_size=resnet_size,
            use_cbam=use_cbam,
            use_film=use_film,
            predict_bboxes=predict_bboxes,
            mlp_organ=mlp_organ,
            num_classes=num_classes_cls,
            n_organs=n_organs_cls,
        )

        self.segm_net = UNet2DFiLM(
            in_channels=in_channels,
            num_classes=num_classes,
            n_organs=n_organs,
            size=size,
            depth=depth,
        )
        self.ll1 = nn.Linear(512, 2048)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x_cls, x_segm, organ_id):
        cls_feat = self.cls_net.get_feat(x_cls, organ_id)
        cls_feat_ll1 = self.ll1(cls_feat)
        gated_cls_feat_ll1 = torch.tanh(self.alpha) * cls_feat_ll1
        gated_cls_feat_ll1 = gated_cls_feat_ll1.unsqueeze(-1).unsqueeze(-1)
        
        bottleneck_feat, feat_list, pads = self.segm_net.encode(x_segm, organ_id)
        fused_feat = bottleneck_feat + gated_cls_feat_ll1

        logits = self.segm_net.decode(x_segm, organ_id, fused_feat, feat_list, pads)
        return logits
