# BraTS2020-MRI-Brain Segmentaion
The following Repo is my implementation of 3D-U-net for Brain MRI segmentation task. 
The database is BraTS2020 (about 10% of the database) can be downloaded from the following Kaggle link:
https://www.kaggle.com/awsaf49/brats20-dataset-training-validation

The full dataset can be asked in the following competetion link:
http://braintumorsegmentation.org/

## Dataset and Pre-processing
The input to 3D-U-net is 4 modalties of MRI scan: 
1. T1 weighted scan
2. Post-contrast T1-weighted scan (T1-gdb), in the code T1_ce
3. T2 weighted scan
4. FLAIR scan (T2 Fluid Attenuated Inversion Recovery)

Each modality with a size of 240*240*155, everyscan is registred (as viewed by Slicer program).
For every modality, we computed the mean and std for the whole training set (80-20 split) and normalize the data according to Z-score normalization (mean 0 std 1). 
No data augmentation was preformed, since we can't assume that MRI images could be found in clinical settings, with any augmentation method. 

## 3D-U-Net
After every modlity (240x240x155) was filtered to (32x7x7x5) in the "Encoder section" (Not a GAN!), we concatenate all modalties together to (128x7*7x5) continue 
to 256 filters and then inserted to the "Decoder section" until we get the segmentaion size (240x240x155). The structure is very similar to 3D-U-net presented at:
Çiçek, Özgün & Abdulkadir, Ahmed & Lienkamp, Soeren & Brox, Thomas & Ronneberger, Olaf. (2016). 3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation. 

In total, the model contain 6,156,692 trainble parameters, and was written with PyTorch. 

## Log, Loss, GPU
I used simple log to collected the loss for every batch and epoch in both training and validation processes.
I used simple Adam optimizer with learning rate of 0.01, I used bacth size of 2 on GeForce RTX™ 3090 GPU. 
I used Dice score for loss and optimization (see explanation: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient), the loss is the (1-score).

## License
The code is free for any use. 
The code is written by Sharon Haimov.

## Reference
[1] B. H. Menze, A. Jakab, S. Bauer, J. Kalpathy-Cramer, K. Farahani, J. Kirby, et al. "The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)", IEEE Transactions on Medical Imaging 34(10), 1993-2024 (2015) DOI: 10.1109/TMI.2014.2377694
[2] S. Bakas, H. Akbari, A. Sotiras, M. Bilello, M. Rozycki, J.S. Kirby, et al., "Advancing The Cancer Genome Atlas glioma MRI collections with expert segmentation labels and radiomic features", Nature Scientific Data, 4:170117 (2017) DOI: 10.1038/sdata.2017.117
[3] S. Bakas, M. Reyes, A. Jakab, S. Bauer, M. Rempfler, A. Crimi, et al., "Identifying the Best Machine Learning Algorithms for Brain Tumor Segmentation, Progression Assessment, and Overall Survival Prediction in the BRATS Challenge", arXiv preprint arXiv:1811.02629 (2018)


