import os
import cv2
import torch
from torch.utils.data import Dataset

class DentalDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.images = sorted(os.listdir(images_dir))
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        image = cv2.imread(os.path.join(self.images_dir, img_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get corresponding mask
        mask_name = f"mask_{img_name.split('_')[1]}"
        mask = cv2.imread(os.path.join(self.masks_dir, mask_name))
        
        # Convert mask to categorical
        mask_categorized = self.categorize_mask(mask)
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask_categorized)
            image = augmented['image']
            mask_categorized = augmented['mask']
            
        return image, mask_categorized
    
    @staticmethod
    def categorize_mask(mask):
        """Convert RGB mask to categorical"""
        categorical_mask = torch.zeros((mask.shape[0], mask.shape[1]), dtype=torch.long)
        # Background is already 0
        categorical_mask[(mask == [255, 255, 255]).all(axis=2)] = 1  # White teeth
        categorical_mask[(mask == [0, 255, 255]).all(axis=2)] = 2    # Yellow caries
        return categorical_mask
