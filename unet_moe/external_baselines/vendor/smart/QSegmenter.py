import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicConvBlock(nn.Module):
    def __init__(self, in_dim=384, conv_channels=[96, 32, 1], kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.layer_channels = conv_channels
        self.param_shapes = []
        total_params = 0

        for i in range(len(conv_channels) - 1):
            Cin, Cout = conv_channels[i], conv_channels[i + 1]
            w = Cout * Cin * kernel_size * kernel_size
            b = Cout
            self.param_shapes.append((Cout, Cin, kernel_size, kernel_size))
            total_params += w + b

        self.decoder = nn.Sequential(
            nn.Conv1d(in_dim, in_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_dim, total_params, 1)
        )

    def forward(self, pooled, x_input):
        B = x_input.size(0)
        params = self.decoder(pooled).squeeze(-1)
        offset = 0
        x = x_input

        for Cout, Cin, kH, kW in self.param_shapes:
            w_len = Cout * Cin * kH * kW
            b_len = Cout
            weight = params[:, offset:offset + w_len].view(B, Cout, Cin, kH, kW)
            offset += w_len
            bias = params[:, offset:offset + b_len].view(B, Cout)
            offset += b_len

            x_next = []
            for i in range(B):
                xi = x[i].unsqueeze(0)
                wi = weight[i]
                bi = bias[i]
                outi = F.conv2d(xi, wi, bi, padding=kH // 2)
                x_next.append(outi)
            x = torch.cat(x_next, dim=0)

        return x


class QFormerLayer(nn.Module):
    def __init__(self, dim=384, heads=8, mlp_ratio=4.0):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, q, kv):
        q_ = self.norm1(q)
        q = q + self.cross_attn(q_, kv, kv)[0]
        q_ = self.norm2(q)
        q = q + self.self_attn(q_, q_, q_)[0]
        q_ = self.norm3(q)
        q = q + self.ffn(q_)
        return q


class QSegmenterHead(nn.Module):
    def __init__(self, q_dim=384, b_dim=768, num_layers=4,
                 num_query=16, conv_in=96, conv_out=1, kernel_size=3):
        super().__init__()
        self.q_proj = nn.Linear(q_dim, q_dim)
        self.b_proj = nn.Linear(b_dim, q_dim)
        self.qformer_layers = nn.ModuleList([
            QFormerLayer(dim=q_dim, heads=8) for _ in range(num_layers)
        ])
        self.attn_pool = nn.Linear(q_dim, 1)

        # 每组包含 3 个 Conv2d 的参数（w+b）
        self.param_shapes = np.array([[32, 12, 1, 1, 32], [1, 32, 1, 1, 1]])
        self.p1w = 32*12*1*1
        self.p1b = 32
        self.p2w = 1*32*1*1
        self.p2b = 1
        total_params = self.p1w + self.p1b + self.p2w + self.p2b

        self.decoder1 = nn.Sequential(
            nn.Conv1d(q_dim, q_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(q_dim, total_params, kernel_size=1)
        )
        self.decoder2 = nn.Sequential(
            nn.Conv1d(q_dim, q_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(q_dim, total_params, kernel_size=1)
        )

        self.conv_in = conv_in
        self.conv_out = conv_out
        self.kernel_size = kernel_size


    def extract_params_apply_dynamic_conv(self, features, params, param_shapes):

        BS = features.shape[0]
        weights_1 = params[:, :self.p1w].reshape([BS] + param_shapes[0, :-1].tolist())
        bias_1 = params[:, self.p1w: self.p1w+self.p1b]
        weights_2 = params[:, self.p1w+self.p1b: self.p1w+self.p1b+self.p2w].reshape([BS] + param_shapes[1, :-1].tolist())
        bias_2 = params[:, self.p1w+self.p1b+self.p2w:]

        features_next = []
        for b in range(BS):
            feature_b = features[b].unsqueeze(0)
            out_b = F.conv2d(feature_b, weight = weights_1[b], bias = bias_1[b], stride = 1, padding = 0, groups = 8)
            out_b = F.relu(out_b)
            out_b = F.conv2d(out_b, weight=weights_2[b], bias=bias_2[b], stride=1, padding=0, groups=1)
            features_next.append(out_b)

        return torch.cat(features_next, dim=0)


    def forward(self, Fquery, Fbottles, Fouts1, Fouts2):
        B = Fquery.shape[0]
        q = self.q_proj(Fquery)
        kv = self.b_proj(Fbottles)

        for layer in self.qformer_layers:
            q = layer(q, kv)

        attn = torch.softmax(self.attn_pool(q), dim=1)
        fpool = (attn * q).sum(dim=1).unsqueeze(-1)

        # 解码两组参数
        params1 = self.decoder1(fpool).squeeze(-1)
        params2 = self.decoder2(fpool).squeeze(-1)

        segmap1 = self.extract_params_apply_dynamic_conv(Fouts1, params1, self.param_shapes)
        segmap2 = self.extract_params_apply_dynamic_conv(Fouts2, params2, self.param_shapes)

        return segmap1, segmap2


if __name__ == "__main__":
    model = QSegmenterHead()
    Fquery = torch.randn(4, 16, 384)
    Fbottles = torch.randn(4, 49, 768)
    Fouts1 = torch.randn(4, 96, 224, 224)
    Fouts2 = torch.randn(4, 96, 56, 56)

    outs = model(Fquery, Fbottles, Fouts1, Fouts2)

    for out in outs:
        print(out.shape)  # torch.Size([4, 1, 224, 224])
