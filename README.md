An Attention Based Network to Achieve P-arrival picking and First-Motion Determination   
Author: Ji Zhang  
Date: 2022.04.25  
Version 1.0.0  

# APPNET   
## Simultaneous Seismic Phase Picking and Polarity Determination with an Attention-based Neural Network

### This repository contains the codes to train and test the network proposed in:             

`Zhang J, Li Z, Zhang J. Simultaneous Seismic Phase Picking and Polarity Determination with an Attention‐Based Neural Network [J]. Seismological Research Letters, 2023.`
      
------------------------------------------- 
### Installation:

   `pip install -r requirements.txt`

or

   `pip install keras-self-attention`
   
------------------------------------------- 
### Short Description:

The focal mechanism of a small earthquake is difficult to determine, but it plays an important role in understanding the regional stress field. The focal mechanisms of small earthquakes can be obtained by inversion of first-motion polarities. Machine learning can help determine polarities efficiently and accurately. The first-motion polarity determination is inseparable from the accuracy of picking and it highly depends on the latter. We propose a first attention-based network to tackle two tasks of picking and polarity determination with encouraging results. APPNET consists of one simple encoder, one decoder, and one classifier.

------------------------------------------- 
### Dataset:

Data from Southern California Earthquake Data Center. [(SCEDC)](https://scedc.caltech.edu/data/deeplearning.html#picking_polarity)  
Download three hdf5 files   
`scsn_p_2000_2017_6sec_0.5r_pick_train.hdf5`  
`scsn_p_2000_2017_6sec_0.5r_pick_test.hdf5`  
`scsn_p_2000_2017_6sec_0.5r_fm_test.hdf5`  
or  
[Traing_dataset](https://service.scedc.caltech.edu/ftp/Ross_FinalTrainedModels/scsn_p_2000_2017_6sec_0.5r_pick_train.hdf5)
[Validation_dataset](https://service.scedc.caltech.edu/ftp/Ross_FinalTrainedModels/scsn_p_2000_2017_6sec_0.5r_pick_test.hdf5)
[Test_dataset](https://service.scedc.caltech.edu/ftp/Ross_FinalTrainedModels/scsn_p_2000_2017_6sec_0.5r_fm_test.hdf5)

------------------------------------------- 
### Run
Download train dataset, validation data, and test dataset to `./dataset/` file.

` Train`
>     python APP_Run.py --mode='train'

`Test`
>     python APP_Run.py --mode='test' --plot_figure

`Predict`
>     python APP_Run.py --mode='predict'  

------------------------------------------- 



   




 





   

