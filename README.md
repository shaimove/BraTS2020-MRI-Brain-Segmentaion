# BraTS2020-MRI-Brain Segmentation
The following Repo is my implementation of 3D-U-net for Brain MRI segmentation task. 
The database is BraTS2020 (about 10% of the database) can be downloaded from the following Kaggle link:
https://www.kaggle.com/awsaf49/brats20-dataset-training-validation

The full dataset can be asked in the following competition link:
http://braintumorsegmentation.org/

## Data and Task
The input to 3D-U-net is 4 modalities of MRI scan: 
1. T1 weighted scan
2. Post-contrast T1-weighted scan (T1-gdb), in the code T1_ce
3. T2 weighted scan
4. FLAIR scan (T2 Fluid Attenuated Inversion Recovery)

The output was annotations of different kind of brain tissues:
0 - Everything else,
1 - necrotic (NCR) and the non-enhancing (NET) tumor core,
2 - peritumoral edema (ED),
4 - enhancing tumor (ET)

see an example for T1-weighted, T2-weighted, and FLAIR images. 
![Image 1](https://github.com/shaimove/BraTS2020-MRI-Brain-Segmentaion/blob/main/Images/modalities.jpg)

see an example of the difference between T1-weighted and T1-weighted with contrast:
![Image 2](https://github.com/shaimove/BraTS2020-MRI-Brain-Segmentaion/blob/main/Images/modalities.jpg)

see an example for the different annotations:
![Image 3](https://github.com/shaimove/BraTS2020-MRI-Brain-Segmentaion/blob/main/Images/annotations.png)


## 3D-U-Net
After every modality (240x240x155) was filtered to (32x7x7x5) in the "Encoder section" (Not a GAN!), I concatenate all modalities together to (128x7*7x5) continue 
to 256 filters and then inserted them to the "Decoder section" until we get the segmentation size (240x240x155). The structure is very similar to 3D-U-net presented at:
Çiçek, Özgün & Abdulkadir, Ahmed & Lienkamp, Soeren & Brox, Thomas & Ronneberger, Olaf. (2016). 3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation. 

In total, the model contains 6,156,692 trainable parameters and was written with PyTorch. 

## Dataset, Log, Loss, GPU
I used a custom dataset class, which loads and transform the images only at training time, I created a utils.py code that stores all the paths to the files, resolutions, and 
pixel statistics to a CSV file, called Training Data Table.csv.
For each modality with a size of 240x240x155, every scan is registered (as viewed by Slicer program).
For every modality, I computed the mean and std for the whole training set (80-20 split) and normalize the data according to Z-score normalization (mean 0 std 1). 
For training 269 examples (with 4 scans + annotations), and 100 validation examples. 
No data augmentation was performed, since I can't assume that MRI images could be found in clinical settings, with any augmentation method. 
I used a simple log to collected the loss for every batch and epoch in both training and validation processes.
I used a simple Adam optimizer with a learning rate of 0.01, I used a batch size of 2 on GeForce RTX™ 3090 GPU. 
I used the Dice score for loss and optimization (see the explanation: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient), the loss is the (1-score).

## License
The code is free for any use. 

## Contact Information
This repo and code were written by Sharon Haimov, Research Engineer at Lumenis ltd. 
email: sharon.haimov@lumenis.com or shaimove@gmail.com

## Reference
[1] B. H. Menze, A. Jakab, S. Bauer, J. Kalpathy-Cramer, K. Farahani, J. Kirby, et al. "The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)", IEEE Transactions on Medical Imaging 34(10), 1993-2024 (2015) DOI: 10.1109/TMI.2014.2377694
[2] S. Bakas, H. Akbari, A. Sotiras, M. Bilello, M. Rozycki, J.S. Kirby, et al., "Advancing The Cancer Genome Atlas glioma MRI collections with expert segmentation labels and radiomic features", Nature Scientific Data, 4:170117 (2017) DOI: 10.1038/sdata.2017.117
[3] S. Bakas, M. Reyes, A. Jakab, S. Bauer, M. Rempfler, A. Crimi, et al., "Identifying the Best Machine Learning Algorithms for Brain Tumor Segmentation, Progression Assessment, and Overall Survival Prediction in the BRATS Challenge", arXiv preprint arXiv:1811.02629 (2018)


