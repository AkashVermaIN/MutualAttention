import torch
from torch import nn, einsum
from utils.drop_path import DropPath
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
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
        return self.fn(self.norm(x), ** kwargs)
 
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
    def __init__(self, dim, num_patches, heads = 8, dim_head = 64, dropout = 0., args=None):
        super().__init__()
        self.args=args
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)
        self.num_patches = num_patches
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.dim = dim
        self.inner_dim = inner_dim
        self.attend = nn.Softmax(dim = -1)
        initial_threshold = args.tam_t if args is not None else 0.0
        self.threshold = nn.Parameter(torch.tensor(initial_threshold))

        # Defining a temperature parameter (can be learnable or fixed)
        # Higher temperature makes the sigmoid steeper (closer to a hard threshold)
        self.sigmoid_temp = 50.0

        self.last_stats = {}

        self.to_qkv = nn.Linear(self.dim, self.inner_dim * 3, bias = False)
        init_weights(self.to_qkv)
        self.to_out = nn.Sequential(    
            nn.Linear(self.inner_dim, self.dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
            

    def forward(self, x, return_attn=False):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = self.attend(dots) # e, f
        colsum = attn.sum(dim=-2, keepdim=True)


        # We catch these stats inside the forward pass to log them later
        if not return_attn: # Optimization: Don't calc grads for stats if just training
             with torch.no_grad():
                # Variance of columns (High variance = Standard ViT collapse)
                col_var = colsum.var()
                self.last_stats['col_var'] = col_var.detach()

        #epsilon = 1e-4
        if self.args is not None and hasattr(self.args, 'eps'):
            self.epsilon = 10.0 ** -float(self.args.eps)
        else:
            self.epsilon = 1e-8

        self.threshold = self.args.tam_t
        #print(f"Epsilon is {self.epsilon}, Threshold is {threshold}")

        # [CHANGE 2] Create a differentiable soft mask
        # If colsum >> threshold, (colsum - threshold) is positive -> sigmoid goes to 1
        # If colsum << threshold, (colsum - threshold) is negative -> sigmoid goes to 0
        
        soft_mask = torch.sigmoid((colsum - self.threshold) * self.sigmoid_temp)

        # Apply the soft mask
        # When soft_mask is 1, we use the normalized attention
        # When soft_mask is 0, we use the original attention
        normalized_attention = (attn / (colsum + self.epsilon)) * soft_mask + attn * (1 - soft_mask)

        # ... continue with row normalization and output projection ...

        rowsum = normalized_attention.sum(dim=-1, keepdim=True)
        normalized_attention = normalized_attention / rowsum
        #normalized_attention = self.attend(normalized_attention)
        #attn = self.attend(normalized_attention) 

        # [NEW] Calculate Entropy (Sharpness) ---------------------------
        if not return_attn:
            with torch.no_grad():
                # Entropy: Higher = more distributed, Lower = sharper
                # We add 1e-9 to avoid log(0)
                entropy = -(normalized_attention * (normalized_attention + 1e-9).log()).sum(dim=-1).mean()
                self.last_stats['entropy'] = entropy.detach()
        # ---------------------------------------------------------------

        out = einsum('b h i j, b h j d -> b h i d', normalized_attention, v) 
            
        out = rearrange(out, 'b h n d -> b n (h d)')
        if return_attn:
            return self.to_out(out), normalized_attention
        return self.to_out(out)
    
    def flops(self):
        flops = 0
        if not self.is_coord:
            flops += self.dim * self.inner_dim * 3 * (self.num_patches+1)
        else:
            flops += (self.dim+2) * self.inner_dim * 3 * self.num_patches  
            flops += self.dim * self.inner_dim * 3  


class Transformer(nn.Module):
    def __init__(self, dim, num_patches, depth, heads, dim_head, mlp_dim_ratio, dropout = 0., stochastic_depth=0., args=None):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.scale = {}

        for i in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(num_patches, dim, Attention(dim, num_patches, heads = heads, dim_head = dim_head, dropout = dropout, args=args)),
                PreNorm(num_patches, dim, FeedForward(dim, num_patches, dim * mlp_dim_ratio, dropout = dropout))
            ]))            
        self.drop_path = DropPath(stochastic_depth) if stochastic_depth > 0 else nn.Identity()
    
    def forward(self, x, return_attn=False):
        all_attn_maps = [] # Store maps if requested
        for i, (attn, ff) in enumerate(self.layers):
            if return_attn:
                # [NEW] Unpack the tuple from Attention
                x_out, attn_map = attn(x, return_attn=True)
                all_attn_maps.append(attn_map)
                x = self.drop_path(x_out) + x
            else:
                x = self.drop_path(attn(x)) + x       
            x = self.drop_path(ff(x)) + x            
            self.scale[str(i)] = attn.fn.scale
        if return_attn:
            return x, all_attn_maps
        return x

class ViT(nn.Module):
    def __init__(self, *, img_size, patch_size, num_classes, dim, depth, heads, mlp_dim_ratio, channels = 3, 
                 dim_head = 16, dropout = 0., emb_dropout = 0., stochastic_depth=0., args=None):
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
        self.transformer = Transformer(self.dim, self.num_patches, depth, heads, dim_head, 
                                        mlp_dim_ratio, dropout, stochastic_depth, args=args)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.num_classes)
        )
        
        self.apply(init_weights)
    def get_patches(self, img):
    	return self.to_patch_embedding(img)
    
    def forward(self, img, return_attn=False):
        # patch embedding
        
        x = self.to_patch_embedding(img)
            
        b, n, _ = x.shape
        
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
      
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        # [NEW] Pass flag to transformer
        if return_attn:
            x, attn_maps = self.transformer(x, return_attn=True)
            return self.mlp_head(x[:, 0]), attn_maps


        x = self.transformer(x)      
        
        return self.mlp_head(x[:, 0])