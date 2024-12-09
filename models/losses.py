import torch
import torch.nn as nn
import torch.nn.functional as F

class ComboLoss(nn.Module):
    def __init__(self, weights=[1.0, 2.0, 4.0]):
        super().__init__()
        self.weights = weights
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()
        
    def forward(self, predictions, target):
        dice = self.dice_loss(predictions, target)
        focal = self.focal_loss(predictions, target)
        return 0.5 * dice + 0.5 * focal

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, predictions, target):
        predictions = F.softmax(predictions, dim=1)
        target_oh = F.one_hot(target, num_classes=predictions.shape[1]).permute(0, 3, 1, 2)
        
        intersection = (predictions * target_oh).sum(dim=(2, 3))
        union = predictions.sum(dim=(2, 3)) + target_oh.sum(dim=(2, 3))
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        
    def forward(self, predictions, target):
        ce_loss = F.cross_entropy(predictions, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss
