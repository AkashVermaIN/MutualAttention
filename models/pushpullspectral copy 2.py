import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math

# -----------------------
# helpers
# -----------------------

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)

# Layer-dependent lambda initialization
def lambda_init_fn(depth):
    return 0.3 + 0.4 * (1 - math.exp(-0.3 * depth))  # safer range [0.3 → ~0.7]

# -----------------------
# Normalization
# -----------------------

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / (norm + self.eps) * self.g

# -----------------------
# Differential Attention
# -----------------------

class SpectralGatedDifferentialAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., depth_idx=0):
        super().__init__()

        assert heads % 2 == 0, "Heads must be even for differential pairing"

        self.dim = dim
        self.heads = heads
        self.half_head = heads // 2
        self.dim_head = dim_head
        self.inner_dim = heads * dim_head
        self.scale = dim_head ** -0.5

        init_lambda = lambda_init_fn(depth_idx)

        # Learnable lambda per head-pair
        self.lambda_q = nn.Parameter(
            torch.full((self.half_head, 1, 1), init_lambda)
        )

        # Learnable sink threshold (positive via softplus)
        self.sink_threshold = nn.Parameter(torch.tensor(1.0))

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, self.inner_dim * 3, bias=False)

        self.diff_norm = RMSNorm(self.inner_dim // 2)

        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim // 2, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, n, _ = x.shape
        h = self.heads

        # Project
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h),
            qkv
        )

        # Split heads
        q1, q2 = q[:, :self.half_head], q[:, self.half_head:]
        k1, k2 = k[:, :self.half_head], k[:, self.half_head:]
        v1 = v[:, :self.half_head]

        # Attention scores
        dots1 = einsum('b h i d, b h j d -> b h i j', q1, k1) * self.scale
        dots2 = einsum('b h i d, b h j d -> b h i j', q2, k2) * self.scale

        attn1 = self.attend(dots1)
        attn2 = self.attend(dots2)

        # ---- Sink suppression (exclude CLS token at index 0) ----
        thresh = F.softplus(self.sink_threshold)

        col_sum = attn1[:, :, 1:, 1:].sum(dim=-2, keepdim=True)
        scale = torch.sigmoid((thresh - col_sum))
        attn1[:, :, 1:, 1:] *= scale
        attn1 = attn1 / (attn1.sum(dim=-1, keepdim=True) + 1e-8)

        # ---- Differential combination ----
        lambda_q = self.lambda_q.unsqueeze(0)  # (1, h, 1, 1)
        attn_diff = attn1 - lambda_q * attn2
        attn_diff = torch.tanh(attn_diff)  # bounded signed attention

        # Aggregate
        out = einsum('b h i j, b h j d -> b h i d', attn_diff, v1)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.diff_norm(out)

        # Residual-safe scaling
        scale = 1 - lambda_q.mean()
        out = out * scale

        return self.to_out(out)

# -----------------------
# PreNorm Wrapper
# -----------------------

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))

# -----------------------
# FeedForward
# -----------------------

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# -----------------------
# Transformer
# -----------------------

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_ratio, dropout, stochastic_depth):
        super().__init__()

        self.layers = nn.ModuleList()
        dpr = torch.linspace(0, stochastic_depth, depth)

        for i in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim,
                    SpectralGatedDifferentialAttention(
                        dim=dim,
                        heads=heads,
                        dim_head=dim_head,
                        dropout=dropout,
                        depth_idx=i
                    )
                ),
                PreNorm(dim,
                    FeedForward(
                        dim=dim,
                        hidden_dim=int(dim * mlp_ratio),
                        dropout=dropout
                    )
                )
            ]))

        self.drop_path = lambda x, p: x if p == 0 else x * (torch.rand_like(x[:, :1]) > p)

    def forward(self, x):
        for attn, ff in self.layers:
            x = x + self.drop_path(attn(x), 0.0)
            x = x + self.drop_path(ff(x), 0.0)
        return x

# -----------------------
# Vision Transformer
# -----------------------

class ViT(nn.Module):
    def __init__(
        self,
        *,
        img_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim_ratio,
        channels=3,
        dim_head=64,
        dropout=0.,
        emb_dropout=0.,
        stochastic_depth=0.
    ):
        super().__init__()

        image_height, image_width = pair(img_size)
        patch_height, patch_width = pair(patch_size)

        self.num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim)
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_ratio=mlp_dim_ratio,
            dropout=dropout,
            stochastic_depth=stochastic_depth
        )

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

        self.apply(init_weights)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding[:, :n + 1]
        x = self.dropout(x)

        x = self.transformer(x)
        return self.mlp_head(x[:, 0])
