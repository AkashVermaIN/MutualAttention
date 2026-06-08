import torch
from torch import nn, einsum
from utils.drop_path import DropPath
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np
import pandas as pd
# helpers
#this is patchwise normalized vit 
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
    def __init__(self, args, dim, num_patches, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)
        self.num_patches = num_patches
        self.forward_counter = 0
        self.epochs = args.epochs
        self.current_epoch = args.current_epoch
        self.debugging = args.debugging
        self.heads = heads
        self.args = args
        self.scale = dim_head ** -0.5
        self.dim = dim
        self.inner_dim = inner_dim
        self.threshold = args.tam_t
        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(self.dim, self.inner_dim * 3, bias = False)
        init_weights(self.to_qkv)
        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, self.dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
            

    def forward(self, x):
        max_forward_counter = 1248
        self.forward_counter += 1
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots) # e, f
        
        if(self.forward_counter >= max_forward_counter and self.debugging == 1):
            #loop through all heads
            for head in range(self.heads):
                presoftmax = dots[0,head,:,:].detach().cpu().numpy()
                postsoftmax = attn[0,head,:,:].detach().cpu().numpy()
                dfpre = pd.DataFrame(presoftmax)
                dfpost = pd.DataFrame(postsoftmax)

                dfpre.to_excel(f"attention_matrix_before_softmax_epoch{self.args.current_epoch+1}_head_{head}.xlsx", index=False)
                dfpost.to_excel(f"attention_matrix_after_softmax_epoch{self.args.current_epoch+1}_head_{head}.xlsx", index=False)

        colsum = attn.sum(dim=-2)
        
        if(self.forward_counter >= max_forward_counter and self.debugging == 1):
            #loop through all heads
            for head in range(self.heads):
                colsumdf = colsum[head].detach().cpu().numpy()
                dfcols = pd.DataFrame(colsumdf)
                dfcols.to_excel(f"sum_of_cols_epoch{self.args.current_epoch+1}_head{head}.xlsx", index=False)


        colsum = attn.sum(dim=-2, keepdim=True)
        

        epsilon = 1e-8
        threshold = self.threshold

        mask = (colsum > threshold).float()
        
        if(self.forward_counter >= max_forward_counter and self.debugging == 1):
            for head in range(self.heads):
                maskdf = mask[0,head].detach().cpu().numpy()
                dfmask = pd.DataFrame(maskdf)
                dfmask.to_excel(f"mask_epoch{self.args.current_epoch+1}_head{head}.xlsx", index=False)

        normalized_attention = attn / (colsum + epsilon) * mask + attn * (1 - mask)
        
        if(self.forward_counter >= max_forward_counter and self.debugging == 1):
            for head in range(self.heads):
                normaldf = normalized_attention[0,head,:,:].detach().cpu().numpy()
                dfnormal = pd.DataFrame(normaldf)
                dfnormal.to_excel(f"normalized_attention_with_threshold_epoch{self.args.current_epoch+1}_head{head}.xlsx", index=False)

        

        attn = normalized_attention

        #rowsumtozeroagain
        attn = self.attend(attn)

            #input("Press any key to continue")
        #print("Counter", self.forward_counter)
        if(self.forward_counter >= max_forward_counter and self.debugging == 1):
            for head in range(self.heads):
                dfattn1 = attn[0,head,:,:].detach().cpu().numpy()
                dfattn1 = pd.DataFrame(dfattn1)
                dfattn1.to_excel(f"final_attention_scoresQK_epoch{self.args.current_epoch+1}_head{head}.xlsx", index=False)







        out = einsum('b h i j, b h j d -> b h i d', attn, v) 

        if(self.forward_counter >= max_forward_counter and self.debugging == 1):
            for head in range(self.heads):
                dfattn2 = out[0,head,:,:].detach().cpu().numpy()
                dfattn2 = pd.DataFrame(dfattn2)
                dfattn2.to_excel(f"final_attention_scoresQKV_epoch{self.args.current_epoch+1}_head{head}.xlsx", index=False)


        out = rearrange(out, 'b h n d -> b n (h d)')

        if(self.forward_counter >= max_forward_counter and self.debugging == 1):
            for head in range(self.heads):
                dfattn = attn[0,head,:,:].detach().cpu().numpy()
                dfattn = pd.DataFrame(dfattn)
                dfattn.to_excel(f"final_attention_scoresQKVRe_epoch{self.args.current_epoch+1}_head{head}.xlsx", index=False)        
        
        if (self.forward_counter == max_forward_counter and self.debugging == 1):
            print("Count Reset", self.forward_counter)
            self.forward_counter = 0
            print(self.forward_counter)
        
        return self.to_out(out)
    

    def flops(self):
        flops = 0
        # Number of tokens (including CLS)
        num_tokens = self.num_patches + 1
        # QKV projections
        flops += self.dim * self.inner_dim * 3 * num_tokens
        # Attention score computation (Q @ K^T)
        flops += self.heads * num_tokens * num_tokens * (self.inner_dim // self.heads)
        # Attention-weighted sum (attn @ V)
        flops += self.heads * num_tokens * num_tokens * (self.inner_dim // self.heads)
        # Final projection (after concatenating heads)
        flops += self.inner_dim * self.dim * num_tokens
        return flops


class Transformer(nn.Module):
    def __init__(self, args, dim, num_patches, depth, heads, dim_head, mlp_dim_ratio, dropout = 0., stochastic_depth=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.scale = {}

        for i in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(num_patches, dim, Attention(args, dim, num_patches, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(num_patches, dim, FeedForward(dim, num_patches, dim * mlp_dim_ratio, dropout = dropout))
            ]))            
        self.drop_path = DropPath(stochastic_depth) if stochastic_depth > 0 else nn.Identity()
    
    def forward(self, x):
        for i, (attn, ff) in enumerate(self.layers):       
            x = self.drop_path(attn(x)) + x
            x = self.drop_path(ff(x)) + x            
            self.scale[str(i)] = attn.fn.scale
        return x

class ViT(nn.Module):
    def __init__(self, *, img_size, patch_size, num_classes, args, dim, depth, heads, mlp_dim_ratio, channels = 3, 
                 dim_head = 16, dropout = 0., emb_dropout = 0., stochastic_depth=0.):
        super().__init__()
        image_height, image_width = pair(img_size)
        patch_height, patch_width = pair(patch_size)
        self.num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.patch_dim = channels * patch_height * patch_width
        self.dim = dim
        self.args = args
        self.num_classes = num_classes

        self.to_patch_embedding = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
                nn.Linear(self.patch_dim, self.dim)
            )
            
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, self.dim))
            
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(args, self.dim, self.num_patches, depth, heads, dim_head, 
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

