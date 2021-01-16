# MetricAndLoss.py
import torch

def DiceLoss(output, target):
    # Define epsilon to avoid divition by zero
    epsilon = 0.01
    
    # calculate the sums of target and output
    target_sum = target.sum(dim=[0,1,2,3,4])
    Prediction_sum = output.sum(dim=[0,1,2,3,4])
    
    # calculate the intersecion between target and output
    correct = (target * output).sum(dim=[0,1,2,3,4])
    
    # Dice score: (2*intersection)/(target + output)
    DiceRatio = (2 * correct + epsilon) / (Prediction_sum + target_sum + epsilon)
    
    # loss is 1-score
    DiceLoss = 1 - DiceRatio
    
    return DiceLoss
    
    
    