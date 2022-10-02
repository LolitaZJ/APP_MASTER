#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 13:25:19 2022

@author: zhangj2
"""
# In[]
import math
import h5py
import matplotlib.pyplot as plt
import numpy as np

import os
import numpy as np
from scipy import signal
import math
try:
    from keras.utils import Sequence
except:
    from tensorflow.keras.utils import Sequence

from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
from obspy.signal.trigger import recursive_sta_lta,classic_sta_lta,trigger_onset
from scipy import interpolate
import random

# In[] 
##calculate cM
def confusion_matrix(pred2,test_label2,file_path=None,name='data'):
    true_up=[i for i in range(len(test_label2)) if np.argmax(test_label2[i])==0]
    true_dn=[i for i in range(len(test_label2)) if np.argmax(test_label2[i])==1]
    true_uw=[i for i in range(len(test_label2)) if np.argmax(test_label2[i])==2]    
    print('Up:%d;Down:%d;Unknown:%d'%(len(true_up),len(true_dn),len(true_uw)))
    
    pred_up=[i for i in range(len(pred2)) if np.argmax(pred2[i])==0]
    pred_dn=[i for i in range(len(pred2)) if np.argmax(pred2[i])==1]
    pred_uw=[i for i in range(len(pred2)) if np.argmax(pred2[i])==2]
    
    print('Up:%d;Down:%d;Unknown:%d'%(len(pred_up),len(pred_dn),len(pred_uw)))
    
    tp_up=[i for i in range(len(pred2)) if np.argmax(pred2[i])==0 and np.argmax(test_label2[i])==0 ]
    tp_dn=[i for i in range(len(pred2)) if np.argmax(pred2[i])==1 and np.argmax(test_label2[i])==1 ]
    tp_uw=[i for i in range(len(pred2)) if np.argmax(pred2[i])==2 and np.argmax(test_label2[i])==2 ]
    
    print('Up:%d;Down:%d;Unknown:%d'%(len(tp_up),len(tp_dn),len(tp_uw)))
    
    fp_up=[i for i in range(len(pred2)) if np.argmax(pred2[i])==0 and np.argmax(test_label2[i])!=0 ]
    fp_dn=[i for i in range(len(pred2)) if np.argmax(pred2[i])==1 and np.argmax(test_label2[i])!=1 ]
    fp_uw=[i for i in range(len(pred2)) if np.argmax(pred2[i])==2 and np.argmax(test_label2[i])!=2 ]
    
    print('Up:%d;Down:%d;Unknown:%d'%(len(fp_up),len(fp_dn),len(fp_uw)))
    # Up:96;Down:40;Unknow:112
    
    ffp_up=[i for i in range(len(pred2)) if np.argmax(pred2[i])==0 and np.argmax(test_label2[i])==1 ]
    ffp_dn=[i for i in range(len(pred2)) if np.argmax(pred2[i])==1 and np.argmax(test_label2[i])==0 ]
    # fp_uw=[i for i in range(len(pred2)) if np.argmax(pred2[i])==2 and np.argmax(test_label2[i])!=2 ]
    
    print('Up:%d;Down:%d;Unknown:%d'%(len(ffp_up),len(ffp_dn),len(fp_uw)))
    
    Pre_U = len(tp_up)/(len(tp_up)+len(ffp_up))
    Pre_D = len(tp_dn)/(len(tp_dn)+len(ffp_dn))
    Pre_K = len(tp_uw)/len(pred_uw)
    print('Pre_Up:%.2f;Pre_Down:%.2f;Pre_Unknown:%.2f'%(Pre_U,Pre_D,Pre_K))
    
    Re_U = len(tp_up)/len(true_up)
    Re_D = len(tp_dn)/len(true_dn)
    try:
        Re_K = len(tp_uw)/len(true_uw)
    except:
        Re_K = np.inf
    
    print('Re_Up:%.2f;Re_Down:%.2f;Re_Unknown:%.2f'%(Re_U,Re_D,Re_K))
    
    if file_path:
        f1 = open(file_path,'a+')
        f1.write('======================\n')
        f1.write(name+'\n')
        f1.write('True:\n')
        f1.write('U; D; K\n')
        f1.write('%d %d %d\n'%(len(true_up),len(true_dn),len(true_uw)))
        f1.write('Pred:\n')
        f1.write('U; D; K\n')
        f1.write('%d %d %d\n'%(len(pred_up),len(pred_dn),len(pred_uw)))       
        f1.write('TP_U; TP_D; TP_K\n')
        f1.write('%d %d %d\n'%(len(tp_up),len(tp_dn),len(tp_uw)))
        f1.write('FP_U; FP_D; FP_K\n')
        f1.write('%d %d %d\n'%(len(fp_up),len(fp_dn),len(fp_uw)))
        f1.write('FFP_U; FFP_D \n')
        f1.write('%d %d\n' %(len(ffp_up),len(ffp_dn)))
        f1.write('Pre_U; Pre_D; Pre_K\n')
        f1.write('%.2f %.2f %.2f\n'%(Pre_U,Pre_D,Pre_K))
        f1.write('Re_U; Re_D; Re_K\n')
        f1.write('%.2f %.2f %.2f\n'%(Re_U,Re_D,Re_K))
        f1.write('======================\n')
        f1.close()        
        
    return tp_up,tp_dn,tp_uw,ffp_up,ffp_dn,fp_up,fp_dn,fp_uw


## generate gauss function
def gaussian(x, sigma,  u):
    y = np.exp(-(x - u) ** 2 / (2 * sigma ** 2)) / (sigma * math.sqrt(2 * math.pi))
    return y/np.max(abs(y))   

# In[]
class DataGenerator_PP1_S(Sequence):

    def __init__(self, L,file_num,gaus,
                 batch_size=128,
                 classes=3,
                 time_shift=False, 
                 shuffle=True,indexes=None,new_label=None,augment=False):
        """
        # Arguments
        ---
            file_num: number of files .
            batch_size: . """
        self.L = L 
        self.batch_size = batch_size
        self.file_num=file_num
        self.gaus=gaus
        if indexes is None:
            self.indexes=np.arange(file_num)
        else:
            self.indexes=indexes
        if new_label is None:
            self.flag=0
        else:
            self.flag=1
            self.new_label=new_label
        
        self.shuffle = shuffle
        self.classes= classes
        self.time_shift= time_shift
        self.augment=augment

    def __len__(self):
        """return: steps num of one epoch. """
        
        return self.file_num// self.batch_size

    def __getitem__(self, index):
        """Gets the `index-th` batch.
        ---
        # Arguments
            index: position of the batch in the Sequence.
        # Returns
            A batch data. """

            
        # get batch data inds.
        batch_inds = self.indexes[index *
                                  self.batch_size:(index+1)*self.batch_size]
        # read batch data
        X, Y1 ,Y2= self._read_data(batch_inds)
        return ({'input': X}, {'pk':Y1, 'po':Y2}) 

    def on_epoch_end(self):
        """shuffle data after one epoch. """
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def _add_noise(self,sig,db,k):
        n=6000
        np.random.seed(k)
        noise=np.random.normal(size=(3,n))
        s2=np.sum(sig**2)/len(sig)
        n2=np.sum(noise[2,:]**2)/len(noise[2,:])
        a=(s2/n2/(10**(db/10)))**(0.5)
        noise=noise*a
        return noise
    
    def _bp_filter(self,data,n,n1,n2,dt):
        wn1=n1*2*dt
        wn2=n2*2*dt
        b, a = signal.butter(n, [wn1,wn2], 'bandpass')
        filtedData = signal.filtfilt(b, a, data)
        return filtedData

    def _normal3(self,data):  
        data2=np.zeros((data.shape[0],data.shape[1],data.shape[2]))
        for i in range(data.shape[0]):
            data1=data[i,:,:]
            x_max=np.max(abs(data1))
            if x_max!=0.0:
                data2[i,:,:]=data1/x_max 
        return data2
    def _taper(self,data,n,N):
        nn=len(data)
        if n==1:
            w=math.pi/N
            F0=0.5
            F1=0.5
        elif n==2:
            w=math.pi/N
            F0=0.54
            F1=0.46
        else:
            w=math.pi/N/2
            F0=1
            F1=1
        win=np.ones((nn,1))
        for i in range(N):
            win[i]=(F0-F1*math.cos(w*(i-1)))
        win1=np.flipud(win)
        
        data1=data*win.reshape(win.shape[0],)
        data1=data1*win1.reshape(win1.shape[0],)
        return data1  

    def _encoder(self,lab,classes=2):
        inx=[i for i in range(len(lab)) if lab[i]>0 ]
        lab[inx]=1
        return to_categorical(lab,classes)
 
    
    def _read_data(self, batch_inds):
        """Read a batch data.
        ---
        # Arguments
            batch_files: the file of batch data.

        # Returns
            data: (batch_size, (5000,1,3)).
            label: (batch_size, (5000,1,num)). """
        #------------------------#  
        np.random.seed(0)
        data=[]
        label1=[]
        label2=[]
        
        for k in batch_inds: 
            # L
            dat=self.L['X'][k]
            # pt1=300
            pt1=int((self.L['Y'][k])*100)
            lab2=self.L['fm'][k]
            if self.flag==1:
                lab2=self.new_label[k]
            
            try:
                lab1=np.zeros(np.size(dat,0),)
                lab1[pt1-50:pt1+50]=self.gaus
            except:
                print(np.size(dat,0),k,pt1,int((self.L['Y'][k])*100))
            
            if self.time_shift :
                for time_sf in range(-300,-100,10):
                    data.append(dat[pt1+time_sf:pt1+time_sf+400])
                    label1.append(lab1[pt1+time_sf:pt1+time_sf+400]) 
                    
                    lab3=to_categorical(lab2,self.classes)
                    label2.append(lab3) 
                    
                    if self.augment:
                        data.append(-dat[pt1+time_sf:pt1+time_sf+400])
                        label1.append(lab1[pt1+time_sf:pt1+time_sf+400]) 
                        if lab2==2:
                            lab3=to_categorical(2,self.classes)
                        if lab2==1:
                            lab3=to_categorical(0,self.classes)
                        if lab2==0:
                            lab3=to_categorical(1,self.classes)                            
                        label2.append(lab3)                                      
            else:
                data.append(dat[pt1-200:pt1+200])
                label1.append(lab1[pt1-200:pt1+200]) 
                lab2=to_categorical(lab2,self.classes)
                label2.append(lab2) 
    
            # L1 
            if self.augment:
            
                num=random.randint(2,4) # zj
                x=np.linspace(0,600,600)
                f1=interpolate.interp1d(x,dat,kind='cubic')
                x1=np.linspace(0,600,num*600)
                
                dat=f1(x1)            
                # pt1=300*num
                pt1=int((self.L['Y'][k]*num)*100)
                lab2=self.L['fm'][k] 
                if self.flag==1:
                    lab2=self.new_label[k]            
       
                lab1=np.zeros(np.size(dat,0),)
                lab1[pt1-50:pt1+50]=self.gaus
                
                if self.time_shift :
                    for time_sf in range(-300,-100,10):
                        data.append(dat[pt1+time_sf:pt1+time_sf+400])
                        label1.append(lab1[pt1+time_sf:pt1+time_sf+400]) 
                        
                        lab3=to_categorical(lab2,self.classes)
                        label2.append(lab3) 
                        
                        if self.augment:
                            data.append(-dat[pt1+time_sf:pt1+time_sf+400])
                            label1.append(lab1[pt1+time_sf:pt1+time_sf+400]) 
                            if lab2==2:
                                lab3=to_categorical(2,self.classes)
                            if lab2==1:
                                lab3=to_categorical(0,self.classes)
                            if lab2==0:
                                lab3=to_categorical(1,self.classes)                            
                            label2.append(lab3) 
    
                else:
                    data.append(dat[pt1-200:pt1+200])
                    label1.append(lab1[pt1-200:pt1+200]) 
                    lab2=to_categorical(lab2,self.classes)
                    label2.append(lab2)                
                
        data=np.expand_dims(np.array(data) ,axis=2)   
        label1=np.array(label1) 
        label2=np.array(label2) 
        return data, label1.reshape(-1,400,1) , label2
# In[]
class DataGenerator_PP1_S_test(Sequence):

    def __init__(self, L,file_num,gaus,
                 batch_size=128,
                 classes=3,
                 time_shift=False, 
                 shuffle=True,indexes=None):
        """
        # Arguments
        ---
            file_num: number of files .
            batch_size: . """
        self.L = L 
        self.batch_size = batch_size
        self.file_num=file_num
        self.gaus=gaus
        if indexes is None:
            self.indexes=np.arange(file_num)
        else:
            self.indexes=indexes
        
        self.shuffle = shuffle
        self.classes= classes
        self.time_shift= time_shift

    def __len__(self):
        """return: steps num of one epoch. """
        
        return self.file_num// self.batch_size+1

    def __getitem__(self, index):
        """Gets the `index-th` batch.
        ---
        # Arguments
            index: position of the batch in the Sequence.
        # Returns
            A batch data. """

            
        # get batch data inds.
        batch_inds = self.indexes[index *
                                  self.batch_size:(index+1)*self.batch_size]
        # read batch data
        X, Y1 ,Y2= self._read_data(batch_inds)
        return ({'input': X}, {'pk':Y1, 'po':Y2}) 

    def on_epoch_end(self):
        """shuffle data after one epoch. """
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def _add_noise(self,sig,db,k):
        n=6000
        np.random.seed(k)
        noise=np.random.normal(size=(3,n))
        s2=np.sum(sig**2)/len(sig)
        n2=np.sum(noise[2,:]**2)/len(noise[2,:])
        a=(s2/n2/(10**(db/10)))**(0.5)
        noise=noise*a
        return noise
    
    def _bp_filter(self,data,n,n1,n2,dt):
        wn1=n1*2*dt
        wn2=n2*2*dt
        b, a = signal.butter(n, [wn1,wn2], 'bandpass')
        filtedData = signal.filtfilt(b, a, data)
        return filtedData

    def _normal3(self,data):  
        data2=np.zeros((data.shape[0],data.shape[1],data.shape[2]))
        for i in range(data.shape[0]):
            data1=data[i,:,:]
            x_max=np.max(abs(data1))
            if x_max!=0.0:
                data2[i,:,:]=data1/x_max 
        return data2
    def _taper(self,data,n,N):
        nn=len(data)
        if n==1:
            w=math.pi/N
            F0=0.5
            F1=0.5
        elif n==2:
            w=math.pi/N
            F0=0.54
            F1=0.46
        else:
            w=math.pi/N/2
            F0=1
            F1=1
        win=np.ones((nn,1))
        for i in range(N):
            win[i]=(F0-F1*math.cos(w*(i-1)))
        win1=np.flipud(win)
        
        data1=data*win.reshape(win.shape[0],)
        data1=data1*win1.reshape(win1.shape[0],)
        return data1  

    def _encoder(self,lab,classes=2):
        inx=[i for i in range(len(lab)) if lab[i]>0 ]
        lab[inx]=1
        return to_categorical(lab,classes)
 
    
    def _read_data(self, batch_inds):
        """Read a batch data.
        ---
        # Arguments
            batch_files: the file of batch data.

        # Returns
            data: (batch_size, (5000,1,3)).
            label: (batch_size, (5000,1,num)). """
        #------------------------#  
        np.random.seed(0)
        data=[]
        label1=[]
        label2=[]
        
        for k in batch_inds: 
            # L
            dat=self.L['X'][k]
            pt1=int((self.L['Y'][k])*100)
            lab2=self.L['fm'][k]
            
            try:
                lab1=np.zeros(np.size(dat,0),)
                lab1[pt1-50:pt1+50]=self.gaus
            except:
                print(np.size(dat,0),k,pt1,int((self.L['Y'][k])*100))
            
            if self.time_shift :
                for time_sf in range(-300,-100,10):
                    data.append(dat[pt1+time_sf:pt1+time_sf+400])
                    label1.append(lab1[pt1+time_sf:pt1+time_sf+400]) 
                    
                    lab3=to_categorical(lab2,self.classes)
                    label2.append(lab3) 

            else:
                data.append(dat[pt1-200:pt1+200])
                label1.append(lab1[pt1-200:pt1+200]) 
                lab2=to_categorical(lab2,self.classes)
                label2.append(lab2) 
    
        data=np.expand_dims(np.array(data) ,axis=2)   
        label1=np.array(label1) 
        label2=np.array(label2) 
        return data, label1.reshape(-1,400,1) , label2 
    
# In[]
def gen_test_data_UD(L,batch_inds,classes=3,time_shift=False,dl=False):
    np.random.seed(0)
    data=[]
    label=[]
    info=[]
    
    for k in range(batch_inds): 
        dat=L['X'][k]
        lab=L['Y'][k]
        
        if dl:
            dist=L['dist'][k]
            evids=L['evids'][k]
            mag=L['mag'][k]
            sncls=L['sncls'][k]
            snr=L['snr'][k]
            sncls=str(sncls, encoding='utf-8')       

        if time_shift :
            for time_sf in range(0,200,10):
                data.append(dat[time_sf:time_sf+400])
                lab2=to_categorical(lab,classes)
                label.append(lab2) 
                if dl:
                    info.append([dist,snr,mag,sncls,evids])

        else:
            data.append(dat[100:500])
            lab2=to_categorical(lab,classes)
            label.append(lab2)             
            if dl:
                info.append([dist,evids,snr,sncls])
    data=np.expand_dims(np.array(data),axis=2)  
    label=np.array(label) 
    
    if dl:
        info.append([dist,evids,snr,sncls])
        return data, label, info
    else:      
        return data, label    
    
    
# In[]    
    
def plot_loss(history_callback,save_path=None,model='model'):
    font2 = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 18,
        }

    history_dict=history_callback.history

    loss_value=history_dict['loss']
    val_loss_value=history_dict['val_loss']
    
    loss_pk=history_dict['pk_loss']
    val_loss_pk=history_dict['val_pk_loss']    
    
    loss_po=history_dict['po_loss']
    val_loss_po=history_dict['val_po_loss']      
    
    try:
        acc_pk=history_dict['pk_accuracy']
        val_acc_pk=history_dict['val_pk_accuracy']
        acc_po=history_dict['po_accuracy']
        val_acc_po=history_dict['val_po_accuracy']        
        
    except:
        acc_value=history_dict['accuracy']
        val_acc_value=history_dict['val_accuracy']  

    epochs=range(1,len(acc_pk)+1)
    if not save_path is None:
        np.savez(save_path+'acc_loss_%s'%model,
                 loss=loss_value,val_loss=val_loss_value,
                 loss_pk=loss_pk,val_loss_pk=val_loss_pk,
                 loss_po=loss_po,val_loss_po=val_loss_po,
                 acc_pk=acc_pk,val_acc_pk=val_acc_pk,
                 acc_po=acc_po,val_acc_po=val_acc_po)

    # acc picking
    figure, ax = plt.subplots(figsize=(8,6))
    plt.plot(epochs,acc_pk,'b',label='Training acc of picking')
    plt.plot(epochs,val_acc_pk,'r',label='Validation acc of picking')  
    plt.tick_params(labelsize=15)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    plt.xlabel('Epochs',font2)
    plt.ylabel('Accuracy',font2)
    plt.legend(prop=font2,loc='lower right')
    if not save_path is None:
        plt.savefig(save_path+'ACC_PK_%s.png'%model,dpi=600)
    plt.show()
    
    # acc polarity 
    figure, ax = plt.subplots(figsize=(8,6))
    plt.plot(epochs,acc_po,'b--',label='Training acc of polarity')
    plt.plot(epochs,val_acc_po,'r--',label='Validation acc of polarity')    
    plt.tick_params(labelsize=15)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    plt.xlabel('Epochs',font2)
    plt.ylabel('Accuracy',font2)
    plt.legend(prop=font2,loc='lower right')
    if not save_path is None:
        plt.savefig(save_path+'ACC_PO_%s.png'%model,dpi=600)
    plt.show()    
    
    # loss 
    figure, ax = plt.subplots(figsize=(8,6))
    plt.plot(epochs,loss_value,'b',label='Training loss')
    plt.plot(epochs,val_loss_value,'r',label='Validation loss')
    plt.tick_params(labelsize=15)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    plt.xlabel('Epochs',font2)
    plt.ylabel('Loss',font2)
    plt.legend(prop=font2)
    if not save_path is None:
        plt.savefig(save_path+'Loss_%s.png'%model,dpi=600)    
    plt.show()    
    
    # loss Picking
    figure, ax = plt.subplots(figsize=(8,6))
    plt.plot(epochs,loss_pk,'b--',label='Training loss of picking')
    plt.plot(epochs,val_loss_pk,'r--',label='Validation loss of picking')    
    plt.tick_params(labelsize=15)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    plt.xlabel('Epochs',font2)
    plt.ylabel('Loss',font2)
    plt.legend(prop=font2)
    if not save_path is None:
        plt.savefig(save_path+'Loss_PK_%s.png'%model,dpi=600)    
    plt.show()    
    
    # loss Polarity
    figure, ax = plt.subplots(figsize=(8,6))
    plt.plot(epochs,loss_po,'b-.',label='Training loss of polarity')
    plt.plot(epochs,val_loss_po,'r-.',label='Validation loss of polarity')    
    
    plt.tick_params(labelsize=15)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    plt.xlabel('Epochs',font2)
    plt.ylabel('Loss',font2)
    plt.legend(prop=font2)
    if not save_path is None:
        plt.savefig(save_path+'Loss_PO_%s.png'%model,dpi=600)    
    plt.show()     