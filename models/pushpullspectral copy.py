import torch
from torch import nn, einsum
from utils.drop_path import DropPath
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math

# helpers
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

# --- INNOVATION: Depth-Dependent Initialization ---
# Lambda (noise cancellation factor) decays with depth. 
# Early layers need strong noise cancellation; deeper layers need semantic mixing.
def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / (norm + self.eps) * self.g

class SpectralGatedDifferentialAttention(nn.Module):
    def __init__(self, dim, num_patches, heads=8, dim_head=64, dropout=0., depth_idx=0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.dim = dim
        self.inner_dim = inner_dim
        
        # We need an even number of heads for Differential split (Signal vs Noise pairs)
        assert heads % 2 == 0, f"Number of heads ({heads}) must be even for Differential Attention"
        self.half_head = heads // 2

        # --- Learnable Lambda Vectors ---
        # A vector of shape (half_head, 1, 1) allows each head-pair to learn its own cancellation rate.
        init_val = lambda_init_fn(depth_idx)
        self.lambda_q = nn.Parameter(torch.zeros(self.half_head, 1, 1).fill_(init_val))
        
        # --- Learnable Sink Threshold ---
        self.sink_threshold = nn.Parameter(torch.tensor(2.0))

        self.attend = nn.Softmax(dim=-1)

        self.to_qkv = nn.Linear(self.dim, self.inner_dim * 3, bias=False)
        
        # --- HeadGroup Normalization ---
        # Stabilizes the differential signal before projection
        self.diff_norm = RMSNorm(self.inner_dim) 
        
        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, self.dim), # Corrected: maps full inner_dim back to dim
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        
        # 1. Projection
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        # 2. Split into Signal (Group 1) and Shadow (Group 2)
        q1, q2 = q[:, :self.half_head], q[:, self.half_head:]
        k1, k2 = k[:, :self.half_head], k[:, self.half_head:]
        
        # 3. Handle Value (V) Logic Correctly
        # Instead of discarding v2, we concatenate v1 and v2 along the feature dimension.
        # This allows the differential attention map (h/2) to attend to a richer feature space (2d).
        v1, v2 = v[:, :self.half_head], v[:, self.half_head:]
        v_combined = torch.cat([v1, v2], dim=-1) # Shape: (b, h/2, n, 2*d_head)

        # 4. Compute Raw Attention Scores
        dots1 = einsum('b h i d, b h j d -> b h i j', q1, k1) * self.scale
        dots2 = einsum('b h i d, b h j d -> b h i j', q2, k2) * self.scale

        # --- BRANCH 1: Signal (Reciprocating Attention) ---
        attn1 = self.attend(dots1)
        
        # Reciprocating Logic: Thresholded Column Scaling
        col_sums = attn1.sum(dim=-2) 
        thresh = torch.abs(self.sink_threshold) + 1e-6
        scale_factor = torch.minimum(torch.ones_like(col_sums), thresh / (col_sums + 1e-8))
        attn1 = attn1 * scale_factor.unsqueeze(-2) 
        
        # Re-normalize rows to maintain probability distribution
        attn1 = attn1 / (attn1.sum(dim=-1, keepdim=True) + 1e-8)

        # --- BRANCH 2: Shadow (Standard Attention) ---
        attn2 = self.attend(dots2)

        # --- COMBINATION: Spectral Gated Differential ---
        # Formula: A_diff = A_signal - lambda * A_shadow
        # We broadcast lambda_q (h/2, 1, 1) across (b, h/2, n, n)
        lambda_param = self.lambda_q.unsqueeze(0) # (1, h/2, 1, 1)
        attn_diff = attn1 - lambda_param * attn2
        
        # 5. Aggregation
        # Apply the cleaned attention map to the combined Values
        out = einsum('b h i j, b h j d -> b h i d', attn_diff, v_combined)
        
        # 6. Residual Scaling & Normalization
        # Align statistics using the learnable lambda to prevent magnitude collapse
        out = out * (1 - lambda_param + 1e-4)
        
        # Flatten heads
        out = rearrange(out, 'b h n d -> b n (h d)') # (b, n, inner_dim)
        
        # Apply Group Norm
        out = self.diff_norm(out) 

        return self.to_out(out)

class PreNorm(nn.Module):
    def __init__(self, num_tokens, dim, fn):
        super().__init__()
        self.dim = dim
        self.num_tokens = num_tokens
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, num_patches, hidden_dim, dropout=0.):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_patches = num_patches
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Transformer(nn.Module):
    def __init__(self, dim, num_patches, depth, heads, dim_head, mlp_dim_ratio, dropout=0., stochastic_depth=0.):
        super().__init__()
        self.layers = nn.ModuleList()
        
        for i in range(depth):
            self.layers.append(nn.ModuleList())
            
        self.drop_path = DropPath(stochastic_depth) if stochastic_depth > 0 else nn.Identity()
    
    def forward(self, x):
        for attn, ff in self.layers:
            x = self.drop_path(attn(x)) + x
            x = self.drop_path(ff(x)) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, img_size, patch_size, num_classes, dim, depth, heads, mlp_dim_ratio, channels=3, 
                 dim_head=64, dropout=0., emb_dropout=0., stochastic_depth=0.):
        super().__init__()
        image_height, image_width = pair(img_size)
        patch_height, patch_width = pair(patch_size)
        self.num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.patch_dim = channels * patch_height * patch_width
        self.dim = dim
        self.num_classes = num_classes
       
        self.to_patch_embedding = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
                nn.Linear(self.patch_dim, self.dim)
            )
            
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, self.dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim))
        self.dropout = nn.Dropout(emb_dropout)
        
        self.transformer = Transformer(self.dim, self.num_patches, depth, heads, dim_head, 
                                      mlp_dim_ratio, dropout, stochastic_depth)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.num_classes)
        )
        
        self.apply(init_weights)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        return self.mlp_head(x[:, 0])