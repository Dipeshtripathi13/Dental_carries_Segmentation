import os
from data.augmentation import get_validation_augmentation
from models.unet import UNet
import torch
import cv2
import numpy as np
from train import train

def load_model(checkpoint_path, config):
    model = UNet(encoder=config.ENCODER, num_classes=len(config.CLASSES))
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(config.DEVICE)
    model.eval()
    return model

def predict_mask(model, image, config):
    """Predict segmentation mask for a single image"""
    transform = get_validation_augmentation(config)
    
    # Preprocess image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    augmented = transform(image=image)
    image_tensor = augmented['image'].unsqueeze(0).to(config.DEVICE)
    
    with torch.no_grad():
        output = model(image_tensor)
        pred_mask = output.argmax(dim=1).squeeze().cpu().numpy()
    
    return pred_mask

def create_colored_mask(pred_mask):
    """Convert prediction to colored visualization"""
    # Define colors for each class (BGR format)
    colors = [
        (0, 0, 0),      # Background - Black
        (255, 255, 255), # Teeth - White
        (0, 255, 255)    # Caries - Yellow
    ]
    
    colored_mask = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
    for class_idx, color in enumerate(colors):
        colored_mask[pred_mask == class_idx] = color
    
    return colored_mask

if __name__ == '__main__':
    # Example usage of training
    from config.config import Config
    
    config = Config()
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    train(config)