# MetricAndLoss.py
import torch

def DiceLoss(output, target):
    
    epsilon = 0.01
    target_sum = target.sum(dim=[1,2,3,4])
    Prediction_sum = output.sum(dim=[1,2,3,4])
    correct = (target * output).sum(dim=[1,2,3,4])
    
    DiceRatio = (2 * correct + epsilon) / (Prediction_sum + target_sum + epsilon)
    DiceLoss = 1 - DiceRatio
    
    return DiceLoss
    
    
    