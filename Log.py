# Log.py
import torch
import numpy as np
import matplotlib.pyplot as plt

class SegmentationLoss(object):
    '''
        The following class track the results of training loop for segmentation
        of 3D Brain MRI scans (T1, T1 ce, T2, FLAIR) with target segmentaiton 
        of 4 labels: None, NCR, ED, ET.
        Since the files are very large, we won't sacve the results, only track 
        the loss value for every batch and every epoch, for both training and 
        validation steps. 
    '''
    def __init__(self):
        
        self.running_batch_loss = []
        self.loss_batch = []
        self.loss_epoch = []
        self.epoch = -1
        
        return None
    
    def BatchUpdate(self,epoch,batch,loss):
        '''
        Update batch loss, of dice score. 
        when a new epoch starts, using self.epoch indicator, we store the recorded
        losses and create new list of running losses. 

        Parameters
        ----------
        epoch : integer, 0 to number of epochs-1
        batch : integer, int 0 to number of batchs per epoch-1
        loss : float32 Tensor size 1, loss of Dice score

        '''
        # Step 1: Transform all the data to CPU and Numpy array
        loss = loss.to('cpu').detach().numpy()

        # Step 2: If we started new epoch
        if epoch != self.epoch:
            # we are at a new epoch, store all previous batchs loss
            self.loss_batch.append(self.running_batch_loss)
            
            # restart the running batch loss
            self.running_batch_loss = []

            # update epoch number
            self.epoch = epoch
        
        else:
            # append loss 
            self.running_batch_loss.append(loss)
        
        return self
            
    def EpochUpdate(self,epoch,loss):
        # we only need to update the loss
        self.loss.append(loss)
        return self 
    
    def getLoss(self,epoch=-1):
        return self.loss[epoch]
    

        
        
        
        
        
