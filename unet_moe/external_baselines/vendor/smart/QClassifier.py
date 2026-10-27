import torch
import torch.nn as nn

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



class QClassifierHead(nn.Module):
    def __init__(self, q_dim=384, num_layers=2):
        super().__init__()
        # Map lesion [BS,96,56,56] -> [BS,3136,96]
        self.lesion_conv = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Align all feats to 384 dim
        self.feat1_proj = nn.Linear(96, 384)
        self.feat2_proj = nn.Linear(192, 384)
        self.feat3_proj = nn.Linear(384, 384)
        self.feat4_proj = nn.Identity()  # already 768, downproject
        self.feat4_down = nn.Linear(768, 384)

        # Q-former
        self.qformer_1 = nn.ModuleList([QFormerLayer(dim=q_dim, heads=8) for _ in range(num_layers)])
        self.qformer_2 = nn.ModuleList([QFormerLayer(dim=q_dim, heads=8) for _ in range(num_layers)])
        self.qformer_3 = nn.ModuleList([QFormerLayer(dim=q_dim, heads=8) for _ in range(num_layers)])
        self.qformer_4 = nn.ModuleList([QFormerLayer(dim=q_dim, heads=8) for _ in range(num_layers)])

        # Final fusion + classifier
        self.fusion = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=384, nhead=8, batch_first=True),
            num_layers=2
        )
        self.classifier_c2 = nn.Sequential(
            nn.Linear(384, 48),
            nn.ReLU(True),
            nn.Linear(48, 2),
        )
        self.classifier_c4 = nn.Sequential(
            nn.Linear(384, 48),
            nn.ReLU(True),
            nn.Linear(48, 4),
        )

    def forward(self, features, lesion_map, query):
        feat1, feat2, feat3, feat4 = features
        BS = feat1.size(0)
        q = query.repeat(BS, 1, 1)

        # lesion map -> [BS, 3136, 96]
        lesion_flat = self.lesion_conv(lesion_map)  # [BS, 96, 56, 56]
        lesion_flat = lesion_flat.flatten(2).transpose(1, 2)  # [BS, 3136, 96]

        # fuse feat1 and lesion info
        feat1 = feat1 + lesion_flat  # same shape [BS, 3136, 96]

        # project all to same dim
        feat1 = self.feat1_proj(feat1)  # [BS, 3136, 384]
        feat2 = self.feat2_proj(feat2)  # [BS, 784, 384]
        feat3 = self.feat3_proj(feat3)  # [BS, 196, 384]
        feat4 = self.feat4_down(self.feat4_proj(feat4))  # [BS, 49, 384]

        # stepwise q-former
        for layer in self.qformer_1:
            q = layer(q, feat1)
        for layer in self.qformer_2:
            q = layer(q, feat2)
        for layer in self.qformer_3:
            q = layer(q, feat3)
        for layer in self.qformer_4:
            q = layer(q, feat4)

        # fuse with deepest feature
        fusion_input = torch.cat([feat4, q], dim=1)  # [BS, 49+16, 384]
        fused = self.fusion(fusion_input)  # [BS, 65, 384]

        pooled = fused.mean(dim=1)  # Global average pooling
        out_c2 = self.classifier_c2(pooled)
        out_c4 = self.classifier_c4(pooled)
        return out_c2, out_c4



if __name__ == "__main__":
    BS = 5
    # 输入张量格式举例
    feat1 = torch.randn(BS, 3136, 96)
    feat2 = torch.randn(BS, 784, 192)
    feat3 = torch.randn(BS, 196, 384)
    feat4 = torch.randn(BS, 49, 768)
    lesion_map = torch.randn(BS, 96, 56, 56)
    query = torch.randn(1, 16, 384)

    # 输出为二分类概率
    model = QClassifierHead()
    output_c2, output_c4 = model([feat1, feat2, feat3, feat4], lesion_map, query)
    print(output_c2.shape)
    print(output_c4.shape)
