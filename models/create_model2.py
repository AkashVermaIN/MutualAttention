def create_model(img_size, n_classes, args):
    if args.model == 'mutualvit':
        from .mutualvit import ViT
        patch_size = 4 if img_size == 32 else 8
        model = ViT(img_size=img_size, patch_size = patch_size, num_classes=n_classes, dim=192, 
                    mlp_dim_ratio=2, depth=9, heads=12, dim_head=192//12,
                    stochastic_depth=0.1, args=args)
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
                                patch_size=patch_size, mlp_ratio=mlp_ratio, depths=depths, num_heads=num_heads, num_classes=n_classes, is_SPT=False, is_LSA=False)
    elif args.model == 'deit_tiny':
        import timm
        model = timm.create_model('deit_tiny_patch16_224', pretrained=False, num_classes=n_classes, img_size=img_size)
    elif args.model == 'mutualvit_learnablethres':
        from .mutualvit_learnablethres import ViT
        patch_size = 4 if img_size == 32 else 8
        model = ViT(img_size=img_size, patch_size = patch_size, num_classes=n_classes, dim=192, 
                    mlp_dim_ratio=2, depth=9, heads=12, dim_head=192//12,
                    stochastic_depth=0.1, args=args)
    elif args.model == 'mvitorig':
        from .mvitorig import ViT
        patch_size = 4 if img_size == 32 else 8
        model = ViT(img_size=img_size, patch_size = patch_size, num_classes=n_classes, dim=192, 
                    mlp_dim_ratio=2, depth=9, heads=12, dim_head=192//12,
                    stochastic_depth=0.1)
    elif args.model == 'reciprocal':
        from .reciprocal import ViT
        patch_size = 4 if img_size == 32 else 8
        model = ViT(img_size=img_size, patch_size = patch_size, num_classes=n_classes, dim=192, 
                    mlp_dim_ratio=2, depth=9, heads=12, dim_head=192//12,
                    stochastic_depth=0.1, args=args)
    elif args.model == 'pushpull':
        from .pushpull import ViT
        patch_size = 4 if img_size == 32 else 8
        model = ViT(img_size=img_size, patch_size = patch_size, num_classes=n_classes, dim=192, 
                    mlp_dim_ratio=2, depth=9, heads=12, dim_head=192//12,
                    stochastic_depth=0.1)
    elif args.model == 'pushpullspectral':
        from .pushpullspectral import ViT
        patch_size = 4 if img_size == 32 else 8
        model = ViT(img_size=img_size, patch_size = patch_size, num_classes=n_classes, dim=192, 
                    mlp_dim_ratio=2, depth=9, heads=12, dim_head=192//12,
                    stochastic_depth=0.1)
    elif args.model == 'mvitnocls':
        from .mvitnocls import ViT
        patch_size = 4 if img_size == 32 else 8
        model = ViT(img_size=img_size, patch_size = patch_size, num_classes=n_classes, dim=192, 
                    mlp_dim_ratio=2, depth=9, heads=12, dim_head=192//12,
                    stochastic_depth=0.1)
    elif args.model == 'mvitclsonly':
        from .mvitclsonly import ViT
        patch_size = 4 if img_size == 32 else 8
        model = ViT(img_size=img_size, patch_size = patch_size, num_classes=n_classes, dim=192, 
                    mlp_dim_ratio=2, depth=9, heads=12, dim_head=192//12,
                    stochastic_depth=0.1)
    elif args.model == 'mutualvit_mixed':
        from .mutualvit_mixed import ViT
        patch_size = 4 if img_size == 32 else 8
        model = ViT(args=args, img_size=img_size, patch_size = patch_size, num_classes=n_classes, dim=192, 
                    mlp_dim_ratio=2, depth=9, heads=12, dim_head=192//12,
                    stochastic_depth=0.1)                              
    elif args.model == 'vit':
        from .vit import ViT
        patch_size = 4 if img_size == 32 else 8
        model = ViT(img_size=img_size, patch_size = patch_size, num_classes=n_classes, dim=192, 
                    mlp_dim_ratio=2, depth=9, heads=12, dim_head=192//12,
                    stochastic_depth=0.1)                     
    return model
