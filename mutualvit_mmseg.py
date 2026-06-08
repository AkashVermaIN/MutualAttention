import torch
import torch.nn as nn
from mmseg.registry import MODELS
from mmengine.model import BaseModule
#from models.vit import ViT
from models.mutualvit import ViT

@MODELS.register_module()

#class MutualViTBackbone(BaseModule):
class ViTBackbone(BaseModule):
    def __init__(self, img_size=224, patch_size=16, embed_dim=192, depth=9, heads=12, **kwargs):
        super().__init__(**kwargs)
        
        self.vit = ViT(
            img_size=img_size, 
            patch_size=patch_size, 
            dim=embed_dim,
            mlp_dim_ratio=2, 
            depth=depth, 
            heads=heads, 
            num_classes=100 # Dummy value, we won't use it
        )
    def forward(self, x):
            x = self.vit.to_patch_embedding(x)
            b, n, _ = x.shape
            
            cls_tokens = self.vit.cls_token.expand(b, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            x += self.vit.pos_embedding[:, :(n + 1)]
            x = self.vit.dropout(x)
            
            x = self.vit.transformer(x)
            spatial_tokens = x[:, 1:, :] 
            
            # Reshape (B, N, C) -> (B, C, H, W)
            B, N, C = spatial_tokens.shape
            grid_size = int(N ** 0.5) 
            feature_map = spatial_tokens.transpose(1, 2).contiguous().view(B, C, grid_size, grid_size)
            
            return (feature_map,)