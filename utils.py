# utils.py
import os
import shutil
import random
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
import nibabel as nib

#%% 
def CreateDataTable(mode,flag_exist):
    '''
    This function creata a pandas DataFrame with data about our dataset,
    training or validation. It calculates the resolution, mean voxel value, 
    stadard deviation of voxel value. In addition it save the paths for all files
    from repo folder. For segmentation it calculates only the path and resolution. 
    At the end, the DataFrame is saved into CSV file, so it the begining of the 
    function, if flag_exist=True, we just read the CSV file into DataFrame and return
    
    Parameters
    ----------
    mode : String
        Training or Validation.
    flag_exist : Boolean
        if True, read the CSV file, if Flase create the DataFrame.

    Returns
    -------
    table : DataFrame
        DataFrame with all path and important stats.

    '''
    # If the file already exist, return the dataframe 
    if flag_exist:
        filename = mode[3:-1] + ' Data Table.csv'
        table = pd.read_csv(filename)
        return table
    
    # the CSV file doesn't exist, create one
    # create list of patients/folders to read from
    patient_folders = os.listdir(mode)
    table = pd.DataFrame()
    table['Patient_name'] = patient_folders
    
    # lists of paths for every file
    T1_path = []; T1_ce_path = []; T2_path = []; FLAIR_path = []; seg_path = []
    
    # lists of resolutions for every file 
    T1_res = []; T1_ce_res = []; T2_res = []; FLAIR_res = []; seg_res = []
    
    # lists of mean values for every input file
    T1_mean = []; T1_ce_mean = []; T2_mean = []; FLAIR_mean = [];
    
    # lists of std values for every input file
    T1_std = []; T1_ce_std = []; T2_std = []; FLAIR_std = [];
    

    # for everty patient/folder calculate dataframe
    for i in tqdm(range(len(patient_folders))):
        # patient folder
        patient = patient_folders[i]
        # assign paths to table
        full_path = mode + patient + '/' + patient
        T1_path.append(full_path + '_t1.nii')
        T1_ce_path.append(full_path + '_t1ce.nii')
        T2_path.append(full_path + '_t2.nii')
        FLAIR_path.append(full_path + '_flair.nii')
        seg_path.append(full_path + '_seg.nii')
        
        # read files
        T1 = nib.load(full_path + '_t1.nii').get_fdata()
        T1ce = nib.load(full_path + '_t1ce.nii').get_fdata()
        T2 = nib.load(full_path + '_t2.nii').get_fdata()
        FLAIR = nib.load(full_path + '_flair.nii').get_fdata()
        Seg = nib.load(full_path + '_seg.nii').get_fdata()
        
        # assign resolution values
        T1_res.append(T1.shape)
        T1_ce_res.append(T1ce.shape)
        T2_res.append(T2.shape)
        FLAIR_res.append(FLAIR.shape)
        seg_res.append(Seg.shape)
        
        # assign mean values
        T1_mean.append(np.mean(T1))
        T1_ce_mean.append(np.mean(T1ce))
        T2_mean.append(np.mean(T2))
        FLAIR_mean.append(np.mean(FLAIR))
        
        # assign std values
        T1_std.append(np.std(T1))
        T1_ce_std.append(np.std(T1ce))
        T2_std.append(np.std(T2))
        FLAIR_std.append(np.std(FLAIR))
    
    # assign all paths to a table
    table['T1 path'] = T1_path
    table['T1 ce path'] = T1_ce_path
    table['T2 path'] = T2_path
    table['FLAIR path'] = FLAIR_path
    table['seg path'] = seg_path
    
    # assign all resolution to a table
    table['T1 resolution'] = T1_res
    table['T1 ce resolution'] = T1_ce_res
    table['T2 resolution'] = T2_res
    table['FLAIR resolution'] = FLAIR_res
    table['seg resolution'] = seg_res
    
    # assign all mean values to a table
    table['T1 mean'] = T1_mean
    table['T1 ce mean'] = T1_ce_mean
    table['T2 mean'] = T2_mean
    table['FLAIR mean'] = FLAIR_mean
    
    # assign all std values to a table
    table['T1 std'] = T1_std
    table['T1 ce std'] = T1_ce_std
    table['T2 std'] = T2_std
    table['FLAIR std'] = FLAIR_std
    
    # save the table to CSV file  
    filename = mode[3:-1] + ' Data Table.csv'
    table.to_csv(filename)
   
    
    return table


#%% 

def CalculateStats(tableTraining,flag_saved):
    '''
    This function calculate the Mean and Standard Deviation for the whole
    Training dataset, in order to perform Z-Score normalization as 
    Pre processing step. 
    
    Parameters
    ----------
    tableTraining : panadas DataFrame of the training dataset
        DataFrame of the training dataset.
    flag_saved : Boolean
        if True, take saved value, if not, calculate.

    Returns
    -------
    Dict_Stats: Dictionary 
        4 key: T1, T1ce, T2, FLAIR. for every key, tuple of (mean,std).

    '''
    if flag_saved:
        Dict_Stats_saved = {'T1': (69.41,228.16),
                      'T1ce' : (74.58,250.80),
                      'T2' : (76.65,243.62),
                      'FLAIR' : (41.87,122.41)}
        return Dict_Stats_saved
    
    # calculate the deviation number (n)
    res = tableTraining['T1 resolution'].iloc[0]
    x,y,z = res[1:-1].split(',')
    x,y,z = int(x),int(y),int(z)
    num_voxels = tableTraining.shape[0] * x * y * z
    
    # Calculate mean values, take into account that all file are with the same resolution
    T1_mean = np.mean(tableTraining['T1 mean'].to_numpy())
    T1_ce_mean = np.mean(tableTraining['T1 ce mean'].to_numpy())
    T2_mean = np.mean(tableTraining['T2 mean'].to_numpy())
    FLAIR_mean = np.mean(tableTraining['FLAIR mean'].to_numpy())

    # accumelated sums to calculate std
    T1_acc = 0; T1ce_acc = 0; T2_acc = 0; FLAIR_acc = 0
    
    # acuumalate all sums
    for i in tqdm(range(tableTraining.shape[0])):
        # T1: get path, read file, subtract mean and square and add to running sum
        T1_path = tableTraining['T1 path'].iloc[i]
        T1 = nib.load(T1_path).get_fdata()
        T1_acc += np.sum(np.square(T1 - T1_mean))
        
        # T1ce: get path, read file, subtract mean and square and add to running sum
        T1ce_path = tableTraining['T1 ce path'].iloc[i]
        T1ce = nib.load(T1ce_path).get_fdata()
        T1ce_acc += np.sum(np.square(T1ce - T1_ce_mean))
        
        # T2: get path, read file, subtract mean and square and add to running sum
        T2_path = tableTraining['T2 path'].iloc[i]
        T2 = nib.load(T2_path).get_fdata()
        T2_acc += np.sum(np.square(T2 - T2_mean))
        
        # FLAIR: get path, read file, subtract mean and square and add to running sum
        FLAIR_path = tableTraining['FLAIR path'].iloc[i]
        FLAIR = nib.load(FLAIR_path).get_fdata()
        FLAIR_acc += np.sum(np.square(FLAIR - FLAIR_mean))

    # calculate std
    T1_std = (T1_acc/num_voxels)**0.5
    T1ce_std = (T1ce_acc/num_voxels)**0.5
    T2_std = (T2_acc/num_voxels)**0.5
    FLAIR_std = (FLAIR_acc/num_voxels)**0.5

    # create dictionary
    Dict_Stats = {'T1': (T1_mean,T1_std),
                  'T1ce' : (T1_ce_mean,T1ce_std),
                  'T2' : (T2_mean,T2_std),
                  'FLAIR' : (FLAIR_mean,FLAIR_std)}
    
    return Dict_Stats




           

#%% 
def count_parameters(model):
    num_parmas = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of Parmaters of this Model is: %s parameters' % num_parmas)
    return None


            
