# Dataset.py
import os
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.cuda
from torch.utils.data import Dataset
from torchvision import transforms



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DatasetMRI(Dataset):
    
    def __init__(self,table,dict_stats):
        
        # Table with all paths, mean values, std values, resolution
        self.table = table 
        
        # Dictionary with all normalization data
        self.dict_stats = dict_stats
        
        # create transform
        self.createTransforms()
        
    
    def __len__(self):
        return self.table.shape[0]
    
    def __getitem__(self,idx):
        # get T1 input
        T1_path = self.table['T1 path'].iloc[idx]
        T1 = nib.load(T1_path).get_fdata()
        
        # get T1_ce input
        T1ce_path = self.table['T1 ce path'].iloc[idx]
        T1_ce = nib.load(T1ce_path).get_fdata()
        
        # get T2 input
        T2_path = self.table['T2 path'].iloc[idx]
        T2 = nib.load(T2_path).get_fdata()
        
        # get FLAIR input
        FLAIR_path = self.table['FLAIR path'].iloc[idx]
        FLAIR = nib.load(FLAIR_path).get_fdata()
        
        # get degmentation mask
        mask = self.table['seg path'].iloc[idx]
        mask = self.preprocess_mask(mask)
        
        # Transform to float32 tensor with 4 dimensions
        T1 = torch.from_numpy(T1,dtype=torch.float32).unsqueeze(0).permute(0,3,1,2)
        T1_ce = torch.from_numpy(T1_ce,dtype=torch.float32).unsqueeze(0).permute(0,3,1,2)
        T2 = torch.from_numpy(T2,dtype=torch.float32).unsqueeze(0).permute(0,3,1,2)
        FLAIR = torch.from_numpy(FLAIR,dtype=torch.float32).unsqueeze(0).permute(0,3,1,2)
        mask = torch.from_numpy(mask,dtype=torch.float32).unsqueeze(0).permute(0,3,1,2)
        
        # preform transforms for every type of imaging and mask
        if self.transform:
            T1 = self.transformT1(T1)
            T1_ce = self.transformT1_ce(T1_ce)
            T2 = self.transformT2(T2)
            FLAIR = self.transformFLAIR(FLAIR)
            #mask = self.transformMask(mask)
            
        # create dictionary
        sample = {'T1': T1, 'T1 ce': T1_ce, 'T2': T2, 'FLAIR': FLAIR, 'Label': mask}
        
        
        return sample

    #%% the following function create mask with one-hot-encoding type
    def preprocess_mask(self,mask):
        # expand dim of mask
        mask = np.expand_dims(mask,axis=0)
        
        # label 0 - None
        mask_None = np.zeros(mask.shape)
        mask_None[mask == 0] = 1
        
        # label 1 - NCR & NET
        maskNCR = np.zeros(mask.shape)
        maskNCR[mask == 1] = 1
        
        # label 2 - ED
        maskED = np.zeros(mask.shape)
        maskED[mask == 2] = 1
        
        # label 3 - ET
        maskET = np.zeros(mask.shape)
        maskET[mask == 4] = 1
        
        # Stack the masks: output is 4*240*240*155
        mask = np.stack([mask_None, maskNCR, maskED,maskET],axis=0)
        
        return mask
    
    #%% The following function create transforms
    def createTransforms(self):
        # define Transformation for every type of imaging
        # For T1 : get mean and std
        mean,std = self.dict_stats['T1'][0],self.dict_stats['T1'][0]
        
        self.transformT1 = transforms.Compose([
            transforms.Normalize(mean=[mean],std=[std])])
                
        # For T1 ce : get mean and std
        mean,std = self.dict_stats['T1 ce'][0],self.dict_stats['T1 ce'][0]
        
        self.transformT1_ce = transforms.Compose([
            transforms.Normalize(mean=[mean],std=[std])])
        
        # For T2 : get mean and std
        mean,std = self.dict_stats['T2'][0],self.dict_stats['T2'][0]
        
        self.transformT2 = transforms.Compose([
            transforms.Normalize(mean=[mean],std=[std])])
        
        # For FLAIR : get mean and std
        mean,std = self.dict_stats['FLAIR'][0],self.dict_stats['FLAIR'][0]
        
        self.transformFLAIR = transforms.Compose([
            transforms.Normalize(mean=[mean],std=[std])])
        
        # For mask
        self.transformMask = transforms.Compose([
            transforms.ToTensor()])
        
        return None
    
    #%% The following function get a instance of a scan and present it in 
    def getScan(self,scan_type,idx):
        # read the modality and number of patient
        path = self.table[scan_type].iloc[idx]
        scan = nib.load(path).get_fdata()
        
        return scan
        
        
        
        
        