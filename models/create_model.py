def create_model(img_size, n_classes, args):
    if args.model == 'mutualvit':
        from .mutualvit import ViT
        patch_size = 4 if img_size == 32 else 8
        model = ViT(img_size=img_size, patch_size = patch_size, num_classes=n_classes, dim=192, 
                    mlp_dim_ratio=2, depth=9, heads=12, dim_head=192//12,
                    stochastic_depth=0.1, args=args)
    elif args.model == 'cait':
        from .cait import CaiT
        patch_size = 4 if img_size == 32 else 8
        model = CaiT(img_size = img_size, patch_size = patch_size, num_classes = n_classes, stochastic_depth = args.sd, is_LSA = False, is_SPT = False)
    elif args.model == 'cait_mutual':
        from .cait_mutual import CaiT
        patch_size = 4 if img_size == 32 else 8
        model = CaiT(img_size = img_size, patch_size = patch_size, num_classes = n_classes, stochastic_depth = args.sd, is_LSA = False, is_SPT = False)
    elif args.model == 'swin':
        from .swin import SwinTransformer
        depths = [2, 6, 4]
        num_heads = [3, 6, 12]
        mlp_ratio = 2
        window_size = 4
        patch_size = 2 if img_size == 32 else 4
        model = SwinTransformer(img_size=img_size, window_size=window_size, drop_path_rate=args.sd, 
                                patch_size=patch_size, mlp_ratio=mlp_ratio, depths=depths, num_heads=num_heads, num_classes=n_classes, is_SPT=False, is_LSA=False)
    elif args.model == 'swin_mutual':
        from .swin_mutual import SwinTransformer
        depths = [2, 6, 4]
        num_heads = [3, 6, 12]
        mlp_ratio = 2
        window_size = 4
        patch_size = 2 if img_size == 32 else 4
        model = SwinTransformer(img_size=img_size, window_size=window_size, drop_path_rate=args.sd, 
                                patch_size=patch_size, mlp_ratio=mlp_ratio, depths=depths, num_heads=num_heads, num_classes=n_classes, is_SPT=False, is_LSA=False, eps=args.eps)                            
    elif args.model == 'vit':
        from .vit import ViT
        patch_size = 4 if img_size == 32 else 8
        model = ViT(img_size=img_size, patch_size = patch_size, num_classes=n_classes, dim=192, 
                    mlp_dim_ratio=2, depth=9, heads=12, dim_head=192//12,
                    stochastic_depth=0.1)                     
    return model
