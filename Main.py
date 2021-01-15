# #Main.py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import os
import nibabel as nib
import torch
from torch.utils import data
from torchvision import transforms

import model
import utils
from Dataset import DatasetMRI
from Log import ClassificationLog


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#%% Dataset
# Define folders 
folder_training = '../Training/'
folder_validation = '../Validation/'

# create DataFrames with data about our dataset
tableTraining = utils.CreateDataTable(folder_training,True)

# now, split the tableTraining TO 80-20 split for validation set
num_training = 269
tableValidation = tableTraining.iloc[num_training:]
tableTraining = tableTraining.iloc[:num_training]

# calculate the z-score normalization for every input type
Dict_stats = utils.CalculateStats(tableTraining,True)


#%% Create dataset and data loader
# Define transformation and data augmentation
transform = transforms.Compose([transforms.ToTensor(),
    transforms.Normalize(mean=[mean],std=[std])])

# define batch size
batch_size_train = 1
batch_size_validation = 1

# define dataset and dataloader for training
train_dataset = DatasetMRI(folder_training,classes,'Training',transform=transform)
train_loader = data.DataLoader(train_dataset,batch_size=batch_size_train,shuffle=True)

# define dataset and dataloader for validation
validation_dataset = DatasetMRI(folder_validation,classes,'Validation',transform=transform)
validation_loader = data.DataLoader(train_dataset,batch_size=batch_size_validation,shuffle=True)


#%% Define parameters
# number of epochs
num_epochs = 1

# load model
model = model.MRIModel().to(device)
utils.count_parameters(model)

# send parameters to optimizer
learning_rate = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# define loss function 
criterion = torch.nn.CrossEntropyLoss()

# initiate logs
trainLog = ClassificationLog()
validationLog = ClassificationLog()

#%% Training


for epoch in range(num_epochs):
    ##################
    ### TRAIN LOOP ###
    ##################
    # set the model to train mode
    model.train()
    
    # initiate training loss
    train_loss = 0
    
    for batch in train_loader:
        # get batch images and labels
        inputs = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        # clear the old gradients from optimizer
        optimizer.zero_grad()
        
        # forward pass: feed inputs to the model to get outputs
        linear_output,output = model(inputs)
        
        # calculate the training batch loss
        loss = criterion(output, torch.max(labels, 1)[1])
        
        # backward: perform gradient descent of the loss w.r. to the model params
        loss.backward()
        
        # update the model parameters by performing a single optimization step
        optimizer.step()
        
        # accumulate the training loss
        train_loss += loss.item()
        
        # update training log
        trainLog.BatchUpdate(epoch,output,linear_output,labels)

            
    #######################
    ### VALIDATION LOOP ###
    #######################
    # set the model to eval mode
    model.eval()
    
    # initiate validation loss
    valid_loss = 0
    
    # turn off gradients for validation
    with torch.no_grad():
        for batch in validation_loader:
            # get batch images and labels
            inputs = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            # forward pass
            linear_output,output = model(inputs)
            
            # validation batch loss
            loss = criterion(output, torch.max(labels, 1)[1]) 
            
            # accumulate the valid_loss
            valid_loss += loss.item()
            
            # update validation log
            validationLog.BatchUpdate(epoch,output,linear_output,labels)
                
    #########################
    ## PRINT EPOCH RESULTS ##
    #########################
    train_loss /= len(train_loader)
    valid_loss /= len(validation_loader)
    # update training and validation loss
    trainLog.EpochUpdate(epoch,train_loss)
    validationLog.EpochUpdate(epoch,valid_loss)
    # print results
    print('Epoch: %s/%s: Training loss: %.3f. Validation Loss: %.3f.'
          % (epoch+1,num_epochs,train_loss,valid_loss))
 
    
    
