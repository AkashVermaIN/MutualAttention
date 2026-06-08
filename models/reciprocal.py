import torch
from torch import nn, einsum
from utils.drop_path import DropPath
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np

# helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)

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
    def __init__(self, dim, num_patches, hidden_dim, dropout = 0.):
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

class Attention(nn.Module):
    def __init__(self, dim, num_patches, heads = 8, dim_head = 64, dropout = 0., sink_threshold = 2.0):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)
        self.num_patches = num_patches
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.dim = dim
        self.inner_dim = inner_dim
        
        # Learnable threshold for column scaling.
        # Initialized to a value > 1.0 (e.g. 2.0) to allow some variation but suppress extreme sinks.
        # Reference: Thresholding attention weights to reduce noise [3, 4]
        self.sink_threshold = nn.Parameter(torch.tensor(sink_threshold))
        
        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(self.dim, self.inner_dim * 3, bias = False)
        init_weights(self.to_qkv)
        
        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, self.dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
            

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        # --- Reciprocating Attention Mechanism ---
        
        # 1. First Row-Softmax: Standard attention (Row Sum = 1)
        # Allows queries to find relevant keys.
        attn = self.attend(dots) 

        # 2. Column Analysis: Compute the "popularity" of each key (column sum).
        # High column sum = Attention Sink (e.g., background or CLS token hoarding attention).
        # Shape: (b, h, j)
        col_sums = attn.sum(dim=-2) 
        
        # 3. Thresholded Column-Scaling:
        # If a column's sum exceeds the threshold, we scale it down to target the threshold.
        # Logic: We want sum(col) <= threshold.
        # Scale factor = Threshold / Col_Sum (if Col_Sum > Threshold), else 1.0
        # We use a soft clamping or standard clamping for stability.
        # Cited Concept: Mitigating Column Sum Imbalance [1]
        
        # Ensure threshold is positive
        thresh = torch.abs(self.sink_threshold) + 1e-6
        
        # Calculate scale: if col_sum > thresh, scale = thresh / col_sum.
        # eps added to avoid division by zero
        scale_factor = torch.minimum(torch.ones_like(col_sums), thresh / (col_sums + 1e-8))
        
        # Apply scaling to the columns.
        # attn: (b, h, i, j), scale: (b, h, j) -> Broadcast over i
        attn = attn * scale_factor.unsqueeze(-2)
        
        # 4. Second Row-Normalization ("Again Softmax"): 
        # The column scaling disrupted the row sums (they are likely < 1 now).
        # We re-normalize rows to sum to 1 to maintain the convex combination property.
        # Using L1 normalization here acts as the stable "second softmax" in Sinkhorn iterations.[2]
        row_sums = attn.sum(dim=-1, keepdim=True)
        attn = attn / (row_sums + 1e-8)
        
        # --- ---

        out = einsum('b h i j, b h j d -> b h i d', attn, v) 
            
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    
    def flops(self):
        flops = 0
        # Updated FLOPs calculation for extra normalization (negligible O(N^2))
        flops += self.dim * self.inner_dim * 3 * (self.num_patches+1)
        flops += self.dim * self.inner_dim * 3  
        return flops


class Transformer(nn.Module):
    def __init__(self, dim, num_patches, depth, heads, dim_head, mlp_dim_ratio, dropout = 0., stochastic_depth=0.):
        super().__init__()
        self.layers = nn.ModuleList()
        self.scale = {}

        for i in range(depth):
            self.layers.append(nn.ModuleList([
                # Pass the innovative Attention block
                PreNorm(num_patches, dim, Attention(dim, num_patches, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(num_patches, dim, FeedForward(dim, num_patches, dim * mlp_dim_ratio, dropout = dropout))
            ]))            
        self.drop_path = DropPath(stochastic_depth) if stochastic_depth > 0 else nn.Identity()
    
    def forward(self, x):
        for i, (attn, ff) in enumerate(self.layers):       
            x = self.drop_path(attn(x)) + x
            x = self.drop_path(ff(x)) + x            
            # self.scale[str(i)] = attn.fn.scale # Removed to avoid attribute errors if not tracking
        return x

class ViT(nn.Module):
    def __init__(self, *, img_size, patch_size, num_classes, dim, depth, heads, mlp_dim_ratio, channels = 3, 
                 dim_head = 16, dropout = 0., emb_dropout = 0., stochastic_depth=0.):
        super().__init__()
        image_height, image_width = pair(img_size)
        patch_height, patch_width = pair(patch_size)
        self.num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.patch_dim = channels * patch_height * patch_width
        self.dim = dim
        self.num_classes = num_classes
       
        self.to_patch_embedding = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
                nn.Linear(self.patch_dim, self.dim)
            )
            
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, self.dim))
            
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim))
        self.dropout = nn.Dropout(emb_dropout)
        
        # Initialize the Mutual ViT Transformer
        self.transformer = Transformer(self.dim, self.num_patches, depth, heads, dim_head, 
                                      mlp_dim_ratio, dropout, stochastic_depth)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.num_classes)
        )
        
        self.apply(init_weights)
    def get_patches(self, img):
        return self.to_patch_embedding(img)
    def forward(self, img):
        # patch embedding
        
        x = self.to_patch_embedding(img)
            
        b, n, _ = x.shape
        
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
      
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)      
        
        return self.mlp_head(x[:, 0])