import torch
from torch.utils.data import Subset
import os
from colorama import Fore, Style
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from collections import defaultdict

def datainfo(logger, args):

    if args.dataset == 'CIFAR10':
        print(Fore.YELLOW+'*'*80)
        logger.debug('CIFAR10')
        print('*'*80 + Style.RESET_ALL)
        n_classes = 10
        img_mean, img_std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        img_size = 32        
        
    elif args.dataset == 'CIFAR100':
        print(Fore.YELLOW+'*'*80)
        logger.debug('CIFAR100')
        print('*'*80 + Style.RESET_ALL)
        n_classes = 100
        img_mean, img_std = (0.5070, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762) 
        img_size = 32 
              
    elif args.dataset == 'T-IMNET':
        print(Fore.YELLOW+'*'*80)
        logger.debug('T-IMNET')
        print('*'*80 + Style.RESET_ALL)
        n_classes = 200
        img_mean, img_std = (0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)
        img_size = 64
        
    elif args.dataset == 'OXFORD-PET':
        print(Fore.YELLOW+'*'*80)
        logger.debug('OXFORD-PET')
        print('*'*80 + Style.RESET_ALL)
        n_classes = 37
        # Standard ImageNet statistics are optimal for transfer learning/fine-grained tasks
        img_mean, img_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        img_size = 224 # Fine-grained datasets generally require higher resolution (224x224)
        
    elif args.dataset == 'OXFORD-PET-SEG':
        # Note target_types='segmentation'
        train_dataset = datasets.OxfordIIITPet(
            root=args.data_path, 
            split='trainval', 
            target_types='segmentation',
            download=True
        )
        
    data_info = dict()
    data_info['n_classes'] = n_classes
    data_info['stat'] = (img_mean, img_std)
    data_info['img_size'] = img_size
    
    return data_info

def dataload(args, augmentations, normalize, data_info):
    if args.dataset == 'CIFAR10':
        train_dataset = datasets.CIFAR10(
            root=os.path.join(args.data_path,'cifar-10'), train=True, download=True, transform=augmentations)
        val_dataset = datasets.CIFAR10(
            root=os.path.join(args.data_path,'cifar-10'), train=False, download=True, transform=transforms.Compose([
            transforms.Resize((data_info['img_size'], data_info['img_size'])),
            transforms.ToTensor(),
            *normalize]))
        
    elif args.dataset == 'CIFAR100':
        train_dataset = datasets.CIFAR100(
            root=os.path.join(args.data_path,'cifar-100'), train=True, download=True, transform=augmentations)
        val_dataset = datasets.CIFAR100(
            root=os.path.join(args.data_path,'cifar-100'), train=False, download=True, transform=transforms.Compose([
            transforms.Resize((data_info['img_size'], data_info['img_size'])),
            transforms.ToTensor(),
            *normalize]))
        
    elif args.dataset == 'T-IMNET':
        train_dataset = datasets.ImageFolder(
            root=os.path.join(args.data_path, 'tiny-imagenet-200', 'train'), transform=augmentations)
        val_dataset = datasets.ImageFolder(
            root=os.path.join(args.data_path, 'tiny-imagenet-200', 'val'), 
            transform=transforms.Compose([
            transforms.Resize((data_info['img_size'], data_info['img_size'])), 
            transforms.ToTensor(), 
            *normalize]))
            
    elif args.dataset == 'OXFORD-PET':
        train_dataset = datasets.OxfordIIITPet(
            root=args.data_path, split='trainval', target_types='category', download=True, transform=augmentations)
        val_dataset = datasets.OxfordIIITPet(
            root=args.data_path, split='test', target_types='category', download=True, transform=transforms.Compose([
            # Unlike CIFAR, Oxford images vary in aspect ratio. Forcing a square resize ensures ViT patch embedding works correctly.
            transforms.Resize((data_info['img_size'], data_info['img_size'])),
            transforms.ToTensor(),
            *normalize]))
    
    # 1. Stratified Sampling (Fixed Ratio per class)
    if hasattr(args, 'train_fixed_ratio') and args.train_fixed_ratio < 100.0:
        targets = train_dataset.targets if hasattr(train_dataset, 'targets') else train_dataset._labels
        
        # Group all indices by their class label
        class_indices = defaultdict(list)
        for idx, target in enumerate(targets):
            class_indices[target].append(idx)
            
        stratified_indices = []
        samples_per_class = 0
        
        # Sample exactly the requested percentage from each class
        for target, indices in class_indices.items():
            class_total = len(indices)
            samples_per_class = int((args.train_fixed_ratio / 100.0) * class_total)
            
            # Shuffle indices for this specific class to remain stochastic
            shuffled_positions = torch.randperm(class_total).tolist()
            selected_indices = [indices[i] for i in shuffled_positions[:samples_per_class]]
            
            stratified_indices.extend(selected_indices)
            
        train_dataset = Subset(train_dataset, stratified_indices)
        
        print(Fore.CYAN + f"==> Data Efficiency Mode (Stratified): Using {args.train_fixed_ratio}% of training data "
                          f"({len(stratified_indices)}/{len(targets)} samples | {samples_per_class} per class)." + Style.RESET_ALL)
                          
    # 2. Pure Random Sampling (Original Method)
    elif hasattr(args, 'train_ratio') and args.train_ratio < 100.0:
        total_samples = len(train_dataset)
        subset_size = int((args.train_ratio / 100.0) * total_samples)
        
        # Generate random indices based on the global seed
        indices = torch.randperm(total_samples).tolist()[:subset_size]
        train_dataset = Subset(train_dataset, indices)
        
        print(Fore.CYAN + f"==> Data Efficiency Mode (Random): Using {args.train_ratio}% of training data ({subset_size}/{total_samples} samples)." + Style.RESET_ALL)
        
    return train_dataset, val_dataset