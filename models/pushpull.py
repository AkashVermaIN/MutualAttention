import torch
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F

# --- HELPERS & UTILS ---

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# Added a simple DropPath implementation in case 'utils.drop_path' is missing
class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)

class PreNorm(nn.Module):
    def __init__(self, dim, fn): # Removed num_tokens (unused)
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
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

# --- INNOVATION: Differential Reciprocating Attention ---
class DiffMutualAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., sink_threshold = 2.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        
        # Guard clause: Differential attention requires splitting heads into two groups
        assert heads % 2 == 0, "Number of heads must be even for Differential Attention (Signal/Shadow split)."

        self.heads = heads
        self.scale = dim_head ** -0.5
        
        # 1. Learnable Threshold for the Reciprocating Branch
        self.sink_threshold = nn.Parameter(torch.tensor(sink_threshold))
        
        # 2. Learnable "Noise Cancellation" factor (Lambda)
        # Using a value that starts calculation near 0.5 but ensures positivity
        self.diff_lambda = nn.Parameter(torch.tensor(0.0)) # exp(0) - 1 = 0, maybe start slightly higher?
        # Let's initialize it such that softplus gives ~0.5
        self.diff_lambda.data.fill_(0.5) 

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
            

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        # --- Differential Split ---
        half_head = h // 2
        
        q1, q2 = q[:, :half_head], q[:, half_head:]
        k1, k2 = k[:, :half_head], k[:, half_head:]
        v1, v2 = v[:, :half_head], v[:, half_head:]

        # --- Branch 1: Signal (Reciprocating Mutual Attention) ---
        dots1 = einsum('b h i d, b h j d -> b h i j', q1, k1) * self.scale
        attn1 = self.attend(dots1) 

        # Reciprocating Logic: Suppress Sinks
        col_sums = attn1.sum(dim=-2) 
        thresh = torch.abs(self.sink_threshold) + 1e-6
        # Fix: Ensure shapes match for broadcasting. col_sums is (b, h, j)
        scale_factor = torch.minimum(torch.ones_like(col_sums), thresh / (col_sums + 1e-8))
        attn1 = attn1 * scale_factor.unsqueeze(-2) # Broadcast over rows i
        
        # Re-Normalize Rows
        row_sums = attn1.sum(dim=-1, keepdim=True)
        attn1 = attn1 / (row_sums + 1e-8)

        # --- Branch 2: Shadow (Noise/Sink Capture) ---
        dots2 = einsum('b h i d, b h j d -> b h i j', q2, k2) * self.scale
        attn2 = self.attend(dots2)

        # --- Differential Subtraction ---
        lambda_param = F.softplus(self.diff_lambda)
        
        # Subtract noise map from signal map
        diff_attn = attn1 - lambda_param * attn2
        
        out1 = einsum('b h i j, b h j d -> b h i d', diff_attn, v1)
        
        # Keep Shadow branch context
        out2 = einsum('b h i j, b h j d -> b h i d', attn2, v2)

        out = torch.cat((out1, out2), dim=1) 
            
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, num_patches, depth, heads, dim_head, mlp_dim_ratio, dropout = 0., stochastic_depth=0.):
        super().__init__()
        self.layers = nn.ModuleList([]) # Initialize empty list
        
        # Calculate hidden dim for FF
        mlp_hidden_dim = int(dim * mlp_dim_ratio)

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, DiffMutualAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_hidden_dim, dropout=dropout))
            ]))
            
        self.drop_path = DropPath(stochastic_depth) if stochastic_depth > 0 else nn.Identity()
    
    def forward(self, x):
        for attn, ff in self.layers:       
            x = self.drop_path(attn(x)) + x
            x = self.drop_path(ff(x)) + x            
        return x

class ViT(nn.Module):
    def __init__(self, *, img_size, patch_size, num_classes, dim, depth, heads, mlp_dim_ratio, channels = 3, 
                 dim_head = 16, dropout = 0., emb_dropout = 0., stochastic_depth=0.):
        super().__init__()
        image_height, image_width = pair(img_size)
        patch_height, patch_width = pair(patch_size)
        
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

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
        
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Robust slicing for pos_embedding
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)      
        
        return self.mlp_head(x[:, 0])