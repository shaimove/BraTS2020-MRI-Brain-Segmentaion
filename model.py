# model.py
import torch
from torch import nn as nn

#%% ResNet50

class MRIModel(nn.Module):
    def __init__(self, in_channels=1, first_conv_size=16,f=3,s=2):
        super().__init__()
        
        # Preprocess stage:
        # from (N*1*512*512) to (N*16*256*256)
        
        
        # initialize weights
        self._init_weights()
    
    def _init_weights(self):
        # initiate with Xavier initialization
        for m in self.modules():
            if type(m) in {nn.Conv2d,nn.Linear}:
                # Weight of layers
                nn.init.xavier_normal_(m.weight)
                # if we have bias
                if m.bias is not None:
                    m.bias.data.fill_(0.01)  
                    
            if type(m) in {nn.BatchNorm2d}:
                # Weight of layers
                nn.init.normal_(m.weight)
                # if we have bias
                if m.bias is not None:
                    m.bias.data.fill_(0.01) 
    
    def forward(self, X):
        
        
        
        return linear,output
        


#%% Identity Block

class IdentityBlock(nn.Module):
    # from (N*F3*H*W) to (N*F3,H,W)
    def __init__(self,filters,f):
        super().__init__()
        
        # Retrieve Filters
        F1, F2, F3 = filters
        
        # Define First Main Path
        self.conv1 = nn.Conv2d(F3,F1,kernel_size=1)
        self.bn1 = nn.BatchNorm2d(F1)
        self.relu1 = nn.ReLU(inplace=True)
        
        # Define second Main Path
        self.conv2 = nn.Conv2d(F1,F2,kernel_size=f,padding=f//2)
        self.bn2 = nn.BatchNorm2d(F2)
        self.relu2 = nn.ReLU(inplace=True)
        
        # Define Third Main Path
        self.conv3 = nn.Conv2d(F2,F3,kernel_size=1)
        self.bn3 = nn.BatchNorm2d(F3)
        self.relu3 = nn.ReLU(inplace=True)
        
    def forward(self,X):
        # define shortcut for future
        X_shortcut = X
        
        # First Main Path
        X = self.conv1(X)
        X = self.bn1(X)
        X = self.relu1(X)
        
        # Second Main Path
        X = self.conv2(X)
        X = self.bn2(X)
        X = self.relu2(X)
        
        # Third Main Path
        X = self.conv3(X)
        X = self.bn3(X)
        
        # Add and Activation
        X = torch.add(X, X_shortcut)
        X = self.relu3(X)
        
        return X

#%% Convolutional  Block


class ConvBlock(nn.Module):
    # from (N*input_size*H*W) to (N*F3,H/2,W/2)
    def __init__(self,input_size,filters,f,s):
        super().__init__()
        
        # Retrieve Filters
        F1, F2, F3 = filters
        
        # Define First Main Path
        self.conv1 = nn.Conv2d(input_size,F1,kernel_size=1)
        self.bn1 = nn.BatchNorm2d(F1)
        self.relu1 = nn.ReLU(inplace=True)
        
        # Define second Main Path
        self.conv2 = nn.Conv2d(F1,F2,kernel_size=f,padding=f//2,stride=s)
        self.bn2 = nn.BatchNorm2d(F2)
        self.relu2 = nn.ReLU(inplace=True)
        
        # Define Third Main Path
        self.conv3 = nn.Conv2d(F2,F3,kernel_size=1)
        self.bn3 = nn.BatchNorm2d(F3)
        self.relu3 = nn.ReLU(inplace=True)
        
        # Define Parrallel Path
        self.conv4 = nn.Conv2d(input_size,F3,kernel_size=1,stride=s)
        self.bn4 = nn.BatchNorm2d(F3)
    
    
    def forward(self,X):
        # define shortcut for future
        X_shortcut = X
        
        # First Main Path
        X = self.conv1(X)
        X = self.bn1(X)
        X = self.relu1(X)
        
        # Second Main Path
        X = self.conv2(X)
        X = self.bn2(X)
        X = self.relu2(X)
        
        # Third Main Path
        X = self.conv3(X)
        X = self.bn3(X)
        
        # Parrallel Path
        X_shortcut = self.conv4(X_shortcut)
        X_shortcut = self.bn4(X_shortcut)
        
        # Add and Activation
        X = torch.add(X, X_shortcut)
        X = self.relu3(X)
        
        return X
        
        
        
        
        
        
        
        
        
        
        
        
        