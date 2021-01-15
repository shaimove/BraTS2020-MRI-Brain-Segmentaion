# MetricAndLoss.py
import torch

def DiceLoss(output, target):
    
    epsilon = 1
    target = target.sum(dim=[1,2,3])
    Prediction = output.sum(dim=[1,2,3])
    correct = (target * Prediction).sum(dim=[1,2,3])
    
    DiceRatio = (2 * correct + epsilon) / (Prediction + target + epsilon)
    DiceLoss = 1 - DiceRatio
    
    return DiceLoss
    
    
    