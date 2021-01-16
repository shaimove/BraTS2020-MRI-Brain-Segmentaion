# model.py
import torch
from torch import nn as nn

#%% ResNet50

class MRIModel(nn.Module):
    def __init__(self, in_channels=1,bridge_channels=32,out_channels=4):
        super().__init__()
        
        # Encoder
        # Encoder Block 1: from (N*1*240*240*155) to (N*32*7*7*5)
        self.encoderT1 = EncoderBlock(in_channels)
        
        # Encoder Block 2: from (N*1*240*240*155) to (N*32*7*7*5)
        self.encoderT1_ce = EncoderBlock(in_channels)
        
        # Encoder Block 3: from (N*1*240*240*155) to (N*32*7*7*5)
        self.encoderT2 = EncoderBlock(in_channels)
        
        # Encoder Block 4: from (N*1*240*240*155) to (N*32*7*7*5)
        self.encoderFLAIR = EncoderBlock(in_channels)
        
        # Bridge Block that unite all modules
        self.bridge = BridgeBlock(4*bridge_channels)
        
        # Decoder Block with shortcuts 
        self.decoder = DecoderBlock(8*bridge_channels,bridge_channels)
                
        # conv + softmax to labels
        self.lastconv = nn.Conv3d(bridge_channels/2,out_channels,kernel_size=(3,3,3),padding=1)
        self.softmax = nn.Softmax(dim=1)
        
        
        # initialize weights
        self._init_weights()
    
    def _init_weights(self):
        # initiate with Xavier initialization
        for m in self.modules():
            if type(m) in {nn.Conv3d,nn.ConvTranspose3d}:
                # Weight of layers
                nn.init.xavier_normal_(m.weight)
                # if we have bias
                if m.bias is not None:
                    m.bias.data.fill_(0.01)  
                    
            if type(m) in {nn.BatchNorm3d}:
                # Weight of layers
                nn.init.normal_(m.weight)
                # if we have bias
                if m.bias is not None:
                    m.bias.data.fill_(0.01) 
    
    def forward(self,T1,T1_ce,T2,FLAIR):
        # Run the encoder from all moduales 
        T1_reduced,T1_Shortcuts = self.encoderT1(T1)
        T1_ce_reduced,T1_ce_Shortcuts = self.encoderT1_ce(T1_ce)
        T2_reduced,T2_Shortcuts = self.encoderT2(T2)
        FLAIR_reduced,FLAIR_Shortcuts =  self.encoderFLAIR(FLAIR)
        
        # combine all moduales and run the bridge
        X = torch.cat([T1_reduced,T1_ce_reduced,T2_reduced,FLAIR_reduced], dim=1)
        X = self.bridge(X)
        
        # Run decoder for bridge and shortcuts
        X = self.decoder(X)
        
        # conv + softmax to labels
        X = self.lastconv(X)
        output = self.softmax(X)
        
        return output
        

#%% Encoder Block
class EncoderBlock(nn.Module):
    
    def __init__(self,in_channels=1):
        super().__init__()
        
        # Encoder
        # ConvBlock 1: from (N*1*240*240*155) to (N*2*120*120*78)
        self.block1 = ConvBlock(in_channels,padd_maxpool=(1,0,0))
        
        # ConvBlock 2: from (N*2*120*120*78) to (N*4*60*60*39)
        self.block2 = ConvBlock(2*in_channels,padd_maxpool=(0,0,0))
        
        # ConvBlock 3: from (N*4*60*60*39) to (N*8*30*30*20)
        self.block3 = ConvBlock(4*in_channels,padd_maxpool=(1,0,0))
        
        # ConvBlock 4: from (N*8*30*30*20) to (N*16*15*15*10)
        self.block4 = ConvBlock(8*in_channels,padd_maxpool=(0,0,0))
        
        # ConvBlock 5: from (N*16*15*15*10) to (N*32*7*7*5)
        self.block5 = ConvBlock(16*in_channels,padd_maxpool=(0,0,0))
        
        # Shortcut sizes
        # X_L1 = (N*2*240*240*155), X_L2 = (N*4*120*120*78)
        # X_L3 = (N*8*60*60*39), X_L4 = (N*16*30*30*20)
        # X_L5 = (N*32*15*15*10)
        
    def forward(self,X):
        
        X,X_L1 = self.block1(X)
        X,X_L2 = self.block2(X)
        X,X_L3 = self.block3(X)
        X,X_L4 = self.block4(X)
        X,X_L5 = self.block5(X)
        
        Shortcuts = [X_L1,X_L2,X_L3,X_L4,X_L5]
        
        return X,Shortcuts

#%% Convolutional Block
class ConvBlock(nn.Module):

    def __init__(self,input_size,padd_maxpool = 0):
        super().__init__()
        # C = input_size
        
        # From (N * C * D * H * W) to (N * C * D * H * W)
        self.conv1 = nn.Conv3d(input_size,input_size,kernel_size=(3,3,3),padding=1)
        self.bn1 = nn.BatchNorm3d(input_size)
        self.relu1 = nn.ReLU(inplace=True)
        
        # From (N * C * D * H * W) to (N * 2C * D * H * W)
        self.conv2 = nn.Conv3d(input_size,2*input_size,kernel_size=(3,3,3),padding=1)
        self.bn2 = nn.BatchNorm3d(2*input_size)
        self.relu2 = nn.ReLU(inplace=True)
        
        # MaxPooling layer
        # From (N * 2C * D * H * W) to (N * 2C * D/2(-/+1) * H/2 * W/2)
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2,padding=padd_maxpool)
    
    def forward(self,X):
        
        # First Main Path
        X = self.conv1(X)
        X = self.bn1(X)
        X = self.relu1(X)
        
        # Second Main Path
        X = self.conv2(X)
        X = self.bn2(X)
        X = self.relu2(X)
        
        # max pool
        X_shortcut = X.copy()
        X = self.maxpool(X)
        
        
        return X,X_shortcut
        
#%% Decoder Block        
class DecoderBlock(nn.Module):
    def __init__(self,input_size,shortcut_size):
        super().__init__()
        # input_size = 256
        # shortcut_size = 32
        # Round 1:
        # Upconv: from (N * 256 * 7 * 7 * 5) to (N * 256 * 15 * 15 * 10)
        # Concatenate with all X_L5 to (N * 384 * 15 * 15 * 10)
        # regular convolution to (N * 128 * 15 * 15 * 10)
        self.upconvblock1 = UpConvBlock(input_size,shortcut_size,
                                        output_padding=(1,0,0),
                                        padding=(1,0,0))
        
        # Round 2:
        # Upconv: from (N * 128 * 15 * 15 * 10) to (N * 128 * 30 * 30 * 20)
        # Concatenate with all X_L4 to (N * 192 * 30 * 30 * 20)
        # regular convolution to (N * 64 * 30 * 30 * 20)
        self.upconvblock2 = UpConvBlock(input_size/2,shortcut_size/2,
                                        output_padding=(1,1,1),
                                        padding=(1,1,1))
        
        # Round 3:
        # Upconv: from (N * 64 * 30 * 30 * 20) to (N * 64 * 60 * 60 * 39)
        # Concatenate with all X_L3 to (N * 96 * 60 * 60 * 39)
        # regular convolution to (N * 32 * 60 * 60 * 39)
        self.upconvblock3 = UpConvBlock(input_size/4,shortcut_size/4,
                                        output_padding=(0,1,1),
                                        padding=(1,1,1))
        
        # Round 4:
        # Upconv: from (N * 32 * 60 * 60 * 39) to (N * 32 * 120 * 120 * 78)
        # Concatenate with all X_L2 to (N * 48 * 120 * 120 * 78)
        # regular convolution to (N * 16 * 120 * 120 * 78)
        self.upconvblock4 = UpConvBlock(input_size/8,shortcut_size/8,
                                        output_padding=(1,1,1),
                                        padding=(1,1,1))
        
        # Round 5:
        # Upconv: from (N * 16 * 120 * 120 * 78) to (N * 16 * 240 * 240 * 155)
        # Concatenate with all X_L2 to (N * 24 * 240 * 240 * 155)
        # regular convolution to (N * 4 * 240 * 240 * 55)
        self.upconvblock5 = UpConvBlock(input_size/16,shortcut_size/16,
                                        output_padding=(0,1,1),
                                        padding=(1,1,1))
    
    def forward(self,X,T1_Shortcuts,T1_ce_Shortcuts,T2_Shortcuts,FLAIR_Shortcuts):
        # Round 1: 
        X_shortcut = torch.concat([T1_Shortcuts[4],T1_ce_Shortcuts[4],
                                   T2_Shortcuts[4],FLAIR_Shortcuts[4]],dim=1)
        X = self.upconvblock1(X,X_shortcut)
        
        # Round 2:
        X_shortcut = torch.concat([T1_Shortcuts[3],T1_ce_Shortcuts[3],
                                   T2_Shortcuts[3],FLAIR_Shortcuts[3]],dim=1)
        X = self.upconvblock2(X,X_shortcut)
        
        # Round 3:
        X_shortcut = torch.concat([T1_Shortcuts[2],T1_ce_Shortcuts[2],
                                   T2_Shortcuts[2],FLAIR_Shortcuts[2]],dim=1)
        X = self.upconvblock3(X,X_shortcut)
        
        # Round 4:
        X_shortcut = torch.concat([T1_Shortcuts[1],T1_ce_Shortcuts[1],
                                   T2_Shortcuts[1],FLAIR_Shortcuts[1]],dim=1)
        X = self.upconvblock4(X,X_shortcut)
        
        # Round 5:
        X_shortcut = torch.concat([T1_Shortcuts[0],T1_ce_Shortcuts[0],
                                   T2_Shortcuts[0],FLAIR_Shortcuts[0]],dim=1)
        X = self.upconvblock5(X,X_shortcut)
        
        return X
        
#%% Uo Convolution Block

class UpConvBlock(nn.Module):
    
    def __init__(self,input_size,shortcut_size,output_padding,padding):
        super().__init__()
        
        # upconv
        self.upconv1 = nn.ConvTranspose3d(input_size,input_size,
                                          kernel_size=3,stride=2,
                                          padding = padding,
                                          output_padding=output_padding)
        
        # conv1
        self.conv1 = nn.Conv3d(input_size + shortcut_size*4,shortcut_size*4,
                               kernel_size = 3, padding = 1)
        self.bn1 = nn.BatchNorm3d(shortcut_size*4)
        self.relu1 = nn.ReLU(inplace=True)
        
        # conv2
        self.conv2 = nn.Conv3d(shortcut_size*4,shortcut_size*4,
                               kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm3d(shortcut_size*4)
        self.relu2 = nn.ReLU(inplace=True)
        
    def forward(self,X,X_shortcut):
        # upconv
        X = self.upconv1(X)
        # Concat
        X = torch.concat([X,X_shortcut],dim = 1)
        # conv1
        X = self.conv1(X)
        X = self.bn1(X)
        X = self.relu1(X)
        # conv2
        X = self.conv2(X)
        X = self.bn2(X)
        X = self.relu2(X)
        
        return X

#%% Bridge Block
class BridgeBlock(nn.Module):
    
    def __init__(self,input_size):
        super().__init__()
        
        # Input size: (N*128*32*7*7*5), After all 4 modules cocatenate
        # from (N*128*32*7*7*5) to (N*128*32*7*7*5)
        self.conv1 = nn.Conv3d(input_size,input_size,kernel_size = 3, padding = 1)
        self.bn1 = nn.BatchNorm3d(input_size)
        self.relu1 = nn.ReLU(inplace=True)
        
        # from (N*128*32*7*7*5) to (N*256*32*7*7*5)
        self.conv2 = nn.Conv3d(input_size,2*input_size,kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm3d(2*input_size)
        self.relu2 = nn.ReLU(inplace=True)
        
        
    def forward(self,X):
        # Part 1 
        X = self.conv1(X)
        X = self.bn1(X)
        X = self.relu1(X)
        # Part 2 
        X = self.conv2(X)
        X = self.bn2(X)
        X = self.relu2(X)
        
        return X
    