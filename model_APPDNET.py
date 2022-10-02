#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 13:27:56 2022

@author: zhangj2
"""
import datetime
import keras
from keras import losses
from keras import optimizers
from keras.models import Sequential
from keras.models import Model,load_model
from keras.layers import Input, Dense, Dropout, Flatten,Embedding, LSTM,GRU,Bidirectional
from keras.layers import Conv1D,Conv2D,MaxPooling1D,MaxPooling2D,BatchNormalization,Reshape
from keras.layers import UpSampling1D,AveragePooling1D,AveragePooling2D,TimeDistributed 
from keras_self_attention import SeqSelfAttention
from keras.callbacks import LearningRateScheduler,EarlyStopping,ModelCheckpoint
# In[]
def build_PP_model(time_input=(400,1),clas=3,filter_size=3,num_filter=[16,32,64],num_dense=128):
    
    inp = Input(shape=time_input, name='input')
    # print(num_filter)
    x = Conv1D(num_filter[0], filter_size, padding = 'same', activation = 'relu')(inp)
    
    x = MaxPooling1D(2)(x)
    x = BatchNormalization()(x)
    
    x = Conv1D(num_filter[1], filter_size, padding = 'same', activation = 'relu')(x)
    x = MaxPooling1D(2)(x)
    x = BatchNormalization()(x)
    
    x = Conv1D(num_filter[2], filter_size, padding = 'same', activation = 'relu')(x)
    x = MaxPooling1D(2)(x)
    x = BatchNormalization()(x)
    
    x = keras.layers.LSTM(units=num_filter[2]*2, return_sequences=True)(x)
    
    at_x,wt = SeqSelfAttention(return_attention=True, attention_width= 20,  
                            attention_activation='relu',name='Atten')(x)
    #----------------------#
    x1 = UpSampling1D(2)(at_x)
    x1 = Conv1D(num_filter[2], filter_size, padding = 'same', activation = 'relu')(x1)
    x1 = BatchNormalization()(x1)
    
    x1 = UpSampling1D(2)(x1)
    x1 = Conv1D(num_filter[1], filter_size, padding = 'same', activation = 'relu')(x1)
    x1 = BatchNormalization()(x1)
    
    x1 = UpSampling1D(2)(x1)
    x1 = Conv1D(num_filter[0], filter_size, padding = 'same', activation = 'relu')(x1)
    x1 = BatchNormalization()(x1)
    
    out1 = Conv1D(1, filter_size, padding = 'same', activation = 'sigmoid',name='pk')(x1)
    
    #----------------------#
    x = Flatten()(at_x)
    
    x = Dense(num_dense,activation = 'relu')(x)
    
    out2 = Dense(clas,activation = 'softmax',name='po')(x)
    
    model = Model(inp, [out1,out2])
    
    return model
# In[]
''' 
##build up model
def build_PP_model(time_input=(400,1),clas=3):
    
    inp = Input(shape=time_input, name='input')
    
    x = Conv1D(16, 5, padding = 'same', activation = 'relu')(inp)
    
    x = MaxPooling1D(2)(x)
    x = BatchNormalization()(x)
    
    x = Conv1D(32, 3, padding = 'same', activation = 'relu')(x)
    x = MaxPooling1D(2)(x)
    x = BatchNormalization()(x)
    
    x = Conv1D(64, 3, padding = 'same', activation = 'relu')(x)
    x = MaxPooling1D(2)(x)
    x = BatchNormalization()(x)
    
    x = keras.layers.LSTM(units=128, return_sequences=True)(x)
    
    at_x,wt = SeqSelfAttention(return_attention=True, attention_width= 20,  
                            attention_activation='relu',name='Atten')(x)
    #----------------------#
    x1 = UpSampling1D(2)(at_x)
    x1 = Conv1D(64, 3, padding = 'same', activation = 'relu')(x1)
    x1 = BatchNormalization()(x1)
    
    x1 = UpSampling1D(2)(x1)
    x1 = Conv1D(32, 3, padding = 'same', activation = 'relu')(x1)
    x1 = BatchNormalization()(x1)
    
    x1 = UpSampling1D(2)(x1)
    x1 = Conv1D(16, 3, padding = 'same', activation = 'relu')(x1)
    x1 = BatchNormalization()(x1)
    
    out1 = Conv1D(1, 3, padding = 'same', activation = 'sigmoid',name='pk')(x1)
    
    #----------------------#
    x = Flatten()(at_x)
    
    x = Dense(128,activation = 'relu')(x)
    
    out2 = Dense(clas,activation = 'softmax',name='po')(x)
    
    model = Model(inp, [out1,out2])
    
    return model
'''