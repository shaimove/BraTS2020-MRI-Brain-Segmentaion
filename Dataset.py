# Dataset.py
import os
import cv2
import numpy as np
import torch
import torch.cuda
from torch.utils.data import Dataset
import torch.nn.functional as F

# beacuse the data set is so small, we can read all the images when initiate 
# the dataset
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DatasetMRI(Dataset):
    
    def __init__(self, data_root,classes,mode,transform=None):
        # root folder is training or testing
        self.data_root = data_root 
        
        # mode of dataset
        self.mode = mode
        
        # initite list of images and tumer classes
        self.classes = classes
        
        # one-hot encoding of labels
        self.classes_one_hot = torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        
        # initiate transform
        self.transform = transform
        
        # create list of all images with labels
        self.images = []
        
        # create list of images to read
        for i,sub_class in enumerate(self.classes):
            # choose sub folder
            sub_folder = os.path.join(data_root, sub_class)
            
            # for every image in subfolder
            for img in os.listdir(sub_folder):
                # create image path
                img_path = os.path.join(data_root,sub_folder,img)
                
                # append path and class
                self.images.append([img_path,self.classes_one_hot[i]])
                        
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,idx):
        # get image path and label
        img_path,label = self.images[idx][0],self.images[idx][1]
        
        # read image 
        image = np.array(cv2.imread(img_path,0))
        #image = np.repeat(np.expand_dims(image,2),3,axis=2)
        
        # preform transforms
        if self.transform:
            image = self.transform(image)
        
        # create dictionary
        sample = {'image': image, 'label': label}
        
        
        return sample



