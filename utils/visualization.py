import cv2
import numpy as np
import torch
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import torchvision.transforms as transforms
import os

def generate_heatmaps(model, val_loader, args, save_dir, amount=5):
    """
    Generates EigenCAM heatmaps for TIP paper visualization.
    Specific for the custom ViT architecture provided.
    """
    print("==> Generating Attention Heatmaps (EigenCAM)...")
    model.eval()
    
    # 1. Define Reshape Transform
    # Converts 1D token sequence back to 2D image for visualization
    def reshape_transform(tensor):
        # Drop CLS token (index 0)
        target = tensor[:, 1:, :] 
        
        # Calculate grid size (e.g., 196 tokens -> 14x14)
        height = width = int(target.size(1) ** 0.5)
        
        result = target.reshape(tensor.size(0), height, width, tensor.size(2))
        
        # [B, H, W, C] -> [B, C, H, W]
        result = result.transpose(2, 3).transpose(1, 2)
        return result

    # 2. Select Target Layer (Specific to your vit.py)
    # Your structure: ViT -> Transformer -> layers (ModuleList) -> [PreNorm(Attn), PreNorm(FF)]
    # We target the 'norm' inside the PreNorm wrapper of the Attention mechanism in the last block.
    try:
        # model.transformer.layers[-1] is the last block (a ModuleList)
        # model.transformer.layers[-1][0] is the PreNorm wrapper for Attention
        # model.transformer.layers[-1][0].norm is the LayerNorm we want to visualize
        target_layers = [model.transformer.layers[-1][0].norm]
    except Exception as e:
        print(f"Error finding target layer: {e}")
        return

    # 3. Initialize CAM
    try:
        cam = EigenCAM(model=model, 
                       target_layers=target_layers, 
                       reshape_transform=reshape_transform)
    except Exception as e:
        print(f"Error initializing EigenCAM: {e}")
        return

    # 4. Process images
    save_folder = os.path.join(save_dir, 'visualizations')
    os.makedirs(save_folder, exist_ok=True)
    
    # Get a batch
    images, targets = next(iter(val_loader))
    
    # Select 'amount' images
    images = images[:amount]
    targets = targets[:amount]
    
    if args.gpu is not None:
        images = images.cuda(args.gpu)
    
    # Generate Heatmaps
    grayscale_cams = cam(input_tensor=images, targets=None)

    for i in range(amount):
        img_tensor = images[i].cpu()
        
        # Denormalize for visualization
        # Using standard ImageNet mean/std reversal
        inv_normalize = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )
        rgb_img = inv_normalize(img_tensor).permute(1, 2, 0).numpy()
        rgb_img = np.clip(rgb_img, 0, 1) 
        
        # Overlay heatmap
        visualization = show_cam_on_image(rgb_img, grayscale_cams[i], use_rgb=True)
        
        # Save side-by-side (Original | Heatmap)
        concat_img = np.hstack((np.uint8(rgb_img * 255), visualization))
        
        filename = f"{save_folder}/heatmap_{i}_class_{targets[i].item()}.jpg"
        cv2.imwrite(filename, cv2.cvtColor(concat_img, cv2.COLOR_RGB2BGR))

    print(f"Saved {amount} heatmaps to {save_folder}")