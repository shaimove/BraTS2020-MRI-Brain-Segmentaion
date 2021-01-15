# Log.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay



class ClassificationLog(object):
    '''
        The following class track the results of training loop for classification
        deep neural network, save the output of the network, linear output before 
        softmax operation, labels (ground truth) and loss. 
        The object is save the data for every epoch, and update at the end of 
        every bacth and every epoch.
    '''
    def __init__(self):
        
        self.output = []
        self.linear_output = []
        self.labels = []
        self.loss = []
        self.epoch = -1
        
        return None
    
    def BatchUpdate(self,epoch,output,linear_output,labels):
        '''
        Update all status vectors after every batch. 
        Every epoch is stored in a new element in list, when a new epoch
        Starts, using self.epoch indicator, we append new numpy array to every 
        list, when didn't start a new epoch, we stack the results in row axis
        to the current epoch. 

        Parameters
        ----------
        epoch : integer, 0 to number of epochs
        output : integer numpy array, size (samples,labels)
        linear_outputs : float32 numpy array, size (samples,labels)
        labels : integer numpy array, size (samples,labels)

        '''
        # Step 1: Transform all the data to CPU and Numpy array
        output = output.to('cpu').detach().numpy()
        linear_output = linear_output.to('cpu').detach().numpy()
        labels = labels.to('cpu').detach().numpy()
        
        
        # Step 2: If we started new epoch, than only append new numpy arrays
        if epoch != self.epoch:
            # we are at a new epoch, append instead of stack
            self.output.append(output)
            self.linear_output.append(linear_output)
            self.labels.append(labels)
            
            # update epoch number
            self.epoch = epoch
        
        
        # Step 3: If we already started the epoch, only stack the new data to current data
        else:
            # we only need to stack the results 
            output_new = np.vstack((self.output[epoch],output))
            self.output[epoch] = output_new
        
            linear_output_new = np.vstack((self.linear_output[epoch],linear_output))
            self.linear_output[epoch] = linear_output_new
            
            labels_new = np.vstack((self.labels[epoch],labels))
            self.labels[epoch] = labels_new
        
        return self
            
    def EpochUpdate(self,epoch,loss):
        # we only need to update the loss
        self.loss.append(loss)
        
        return self 
    
    # function to return the output,labels,linear_output,loss
    def getOutput(self,epoch=-1):
        return self.output[epoch]
    
    def getLabels(self,epoch=-1):
        return self.labels[epoch]
    
    def getLinearOutput(self,epoch=-1):
        return self.linear_output[epoch]
        
    def getLoss(self,epoch=-1):
        return self.loss[epoch]
    
    # Plot confusion matrix
    def PlotConfusionMatrix(self,classes,epoch=-1):
        y_true  = np.argmax(self.labels[epoch],axis=1)
        y_pred = np.argmax(self.output[epoch],axis=1)
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(cm)
        
        disp.plot()
        
        
        
        
        
