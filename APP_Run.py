#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 00:00:13 2022

@author: zhangj2
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 11:57:32 2022

APPDNET:
    picking and polarity determination with an attention network
` Train`
>     python APP_Run.py --mode='train' 

`Test`
>     python APP_Run.py --mode='test' --plot_figure

`Predict`
>     python APP_Run.py --mode='predict'  
    
    
@author: zhangj2
"""

# In[]
cuda_kernel='1'
import os
os.getcwd()
import tensorflow as tf

import datetime
import h5py
import matplotlib.pyplot as plt
import numpy as np
from skimage import transform
from scipy import interpolate
import argparse
from utils_APPDNET import gaussian,plot_loss,DataGenerator_PP1_S,DataGenerator_PP1_S_test,confusion_matrix,gen_test_data_UD
from model_APPDNET import build_PP_model
from keras_self_attention import SeqSelfAttention
from keras.callbacks import LearningRateScheduler,EarlyStopping,ModelCheckpoint  
from keras.models import Model,load_model
from keras.utils.np_utils import to_categorical
import random
import pandas as pd

import matplotlib  
matplotlib.use('Agg') 

# In[]
def start_gpu(args):
    cuda_kernel=args.GPU
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_kernel
    # os.environ["CUDA_DEVICE_ORDER"] = args.PCI_BUS_ID
    
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    print('Physical GPU：', len(gpus))
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print('Logical GPU：', len(logical_gpus))




def read_args():

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--GPU",
                        default="0",
                        help="set gpu ids")  
    
    parser.add_argument("--mode",
                        default="train",
                        help="/train/predict/hinet")
    
    parser.add_argument("--model_name",
                        default="APPNET_LOLITA",
                        help="model name")  
    
    parser.add_argument("--epochs",
                        default=100,
                        type=int,
                        help="number of epochs (default: 10)")
    
    parser.add_argument("--batch_size",
                        default=256,
                        type=int,
                        help="batch size")
    
    parser.add_argument("--learning_rate",
                        default=0.001,
                        type=float,
                        help="learning rate")
    
    parser.add_argument("--patience",
                        default=5,
                        type=int,
                        help="early stopping")
    
    parser.add_argument("--clas",
                        default=3,
                        type=int,
                        help="number of class") 
    
    parser.add_argument("--monitor",
                        default="val_loss",
                        help="monitor the val_loss/loss/acc/val_acc")  
    
    parser.add_argument("--monitor_mode",
                        default="min",
                        help="min/max/auto") 
    
    parser.add_argument("--use_multiprocessing",    
                        default=False,
                        help="False/True")     
    
    parser.add_argument("--workers",
                        default=1,
                        type=int,
                        help="workers")     
    
    parser.add_argument("--loss",
                        default=['mse','categorical_crossentropy'],
                        type=list,
                        help="loss fucntion")  
    
    parser.add_argument("--num_filter",
                        default=[16,32,64],
                        type=int,
                        nargs='+',
                        help="num_filter") 
    
    parser.add_argument("--filter_size",
                        default=3,
                        type=int,
                        help="filter_size")      
    
    parser.add_argument("--num_dense",
                        default=128,
                        type=int,
                        help="num_dense")      
       
    parser.add_argument("--model_dir",
                        default='./model/',
                        help="Checkpoint directory (default: None)")

    parser.add_argument("--num_plots",
                        default=10,
                        type=int,
                        help="Plotting trainning results")
    
    parser.add_argument("--input_length",
                        default=400,
                        type=int,
                        help="input length")
        
    parser.add_argument("--data_dir",
                        default=None,
                        help="Input file directory")
    
    parser.add_argument("--data_list",
                        default=None,
                        help="Input csv file")
    
    parser.add_argument("--train_dir",
                        default="./dataset/scsn_p_2000_2017_6sec_0.5r_pick_train.hdf5",
                        help="Input file directory")
    
    parser.add_argument("--train_list",
                        default=None,
                        help="Input csv file")
    
    parser.add_argument("--valid_dir",
                        default='./dataset/scsn_p_2000_2017_6sec_0.5r_pick_test.hdf5',
                        help="Input file directory")
    
    parser.add_argument("--valid_list",
                        default=None,
                        help="Input csv file")
    
    parser.add_argument("--org_dir",
                        default='./dataset/scsn_p_2000_2017_6sec_0.5r_fm_test.hdf5',
                        help="SCSN testing data")    

    
    parser.add_argument("--output_dir",
                        default='./result/',
                        help="Output directory")
    
    parser.add_argument("--conf_dir",
                        default='./model_configure/',
                        help="Configure directory")  
    
    parser.add_argument("--acc_loss_fig",
                        default='./acc_loss_fig/',
                        help="acc&loss directory")    
    
    parser.add_argument("--time_shift",
                        default=True,
                        help="False/True")
    
    parser.add_argument("--shuffle",
                        default=True,
                        help="False/True")
    
    parser.add_argument("--augment",
                        default=True,
                        help="False/True")       
    
    parser.add_argument("--plot_figure",
                        action="store_true",
                        help="If plot figure for test")
    
    parser.add_argument("--save_result",
                        action="store_true",
                        help="If save result for test")
      
    parser.add_argument("--exam",
                        default='example1',
                        help="example1/example2")  
        
    args = parser.parse_args()
    return args


# In[]Parameters of network
def set_configure(args):
    model_name=args.model_name
    time_input=(args.input_length,1)
    epochs=args.epochs
    patience=args.patience
    monitor=args.monitor
    mode=args.monitor_mode
    batch_size=args.batch_size
    num_filter=args.num_filter,
    filter_size=args.filter_size,
    num_dense=args.num_dense,

    clas=args.clas
    loss=args.loss


    if not os.path.exists(args.conf_dir):
        os.mkdir(args.conf_dir)

    # save configure
    f1 = open(args.conf_dir+'Conf_%s.txt'%args.model_name,'w')
    f1.write('Model: %s'%model_name+'\n')
    f1.write('num_filter: %s'%num_filter+'\n')
    f1.write('filter_size: %d'%filter_size+'\n')
    f1.write('num_dense: %d'%num_dense+'\n')
    f1.write('epochs: %d'%epochs+'\n')
    f1.write('batch_size: %d'%batch_size+'\n')
    f1.write('monitor: %s'%monitor+'\n')
    f1.write('mode: %s'%mode+'\n')
    f1.write('patience: %d'%patience+'\n')
    f1.write('time_input: %s'%str(time_input)+'\n')
    f1.write('class: %d'%clas+'\n')
    f1.write('loss: %s'%loss+'\n')
    

    f1.close()
    

def main(args):
    time_input=(args.input_length,1)
    gaus=gaussian(np.linspace(-5, 5, 100),1,0) 
    
    if args.mode=='train':
        args.clas=3
        clas=args.clas 
        #=======train_dataset======#
        print('load training data')
        f22=h5py.File(args.train_dir,'r')
        # train_index=[i for i in range(len(f22['fm'])) if f22['fm'][i]<3 ]
        # file_num=len(train_index)
        file_num=len(f22['fm'][:])
        steps_per_epoch=file_num//args.batch_size
        train_generator=DataGenerator_PP1_S(f22,file_num,gaus,batch_size=args.batch_size, augment=True,
                                            classes=clas,time_shift=True, shuffle=True)#,indexes=train_index)  
        #=======validation_dataset======#
        print('load validation data')
        f44=h5py.File(args.valid_dir,'r')
        # test_index=[i for i in range(len(f44['fm'])) if f44['fm'][i]<3 ]
        # file_num1=len(test_index)
        file_num1=len(f44['fm'][:])
        test_index=[i for i in range(len(f44['fm']))]
        np.random.seed(0)
        np.random.shuffle(test_index)
        file_num1=10240
        validation_steps=file_num1//args.batch_size
        validation_generator=DataGenerator_PP1_S(f44,file_num1,gaus,batch_size=args.batch_size, 
                                                 classes=clas,time_shift=False, shuffle=False,indexes=test_index[:file_num1]) 

   
    if args.mode=='train':
        # model=build_PP_model(time_input=time_input,clas=clas)
        model=build_PP_model(time_input=time_input,clas=clas,num_filter=args.num_filter,
                             filter_size=args.filter_size,num_dense=args.num_dense)
        print(args.mode)
        if not os.path.exists(args.model_dir):
            os.mkdir(args.model_dir)        
        model.compile(loss=args.loss,optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),metrics=['accuracy'])
        saveBestModel= ModelCheckpoint(args.model_dir+'%s.h5'%args.model_name, monitor=args.monitor,
                                       verbose=1, save_best_only=True,mode=args.monitor_mode)
        estop = EarlyStopping(monitor=args.monitor, patience=args.patience, verbose=0, mode=args.monitor_mode)
        callbacks_list = [saveBestModel,estop]
        # fit
        begin = datetime.datetime.now()
        history_callback=model.fit_generator(generator=train_generator,    
                                    steps_per_epoch=steps_per_epoch,
                                            epochs=args.epochs, 
                                              verbose=1,
                                              callbacks=callbacks_list, 
                                              use_multiprocessing=args.use_multiprocessing,
                                              workers=args.workers,                                              
                                     validation_data=validation_generator,
                                     validation_steps=validation_steps)    
        
        model.save_weights(args.model_dir+'%s_wt.h5'%args.model_name) 
        end = datetime.datetime.now()

        #=======plot acc & loss======#
        print('plot acc & loss')
        if not os.path.exists(args.acc_loss_fig):
            os.mkdir(args.acc_loss_fig)
        plot_loss(history_callback,save_path=args.acc_loss_fig,model=args.model_name)  
              
    elif args.mode=='test':
        print(args.mode)
        clas=args.clas
        ## load model 
        try:
            model=load_model(args.model_dir+args.model_name+'.h5',custom_objects=SeqSelfAttention.get_custom_objects())
            # model.save_weights(args.model_dir+'%s_wt.h5'%args.model_name)
        except:
            print('Do not exists model!')
            
        #===========QC=================#
        f44=h5py.File(args.valid_dir,'r')
        file_num1=len(f44['Y'][:])
        validation_generator2=DataGenerator_PP1_S(f44,file_num1,gaus,batch_size=file_num1, 
                                          classes=clas,time_shift=False, shuffle=False,
                                          indexes=None,new_label=None,augment=False)         
        begin = datetime.datetime.now()
        gen=iter(validation_generator2)
        tmp=next(gen)
        test_data=tmp[0]['input']
        test_label1=tmp[1]['pk']
        test_label2=tmp[1]['po']
        pred1,pred2=model.predict(test_data)
        end = datetime.datetime.now()
        print('Testing time:',end-begin)        
        
        ## QC time
        tp_t=np.argmax(pred1[:,:,0],axis=1)
        tp_tr=np.argmax(test_label1[:,:,0],axis=1)
        dt_p=tp_tr-tp_t
        print('MAE: %.2f (s)'%np.mean(abs(dt_p*0.01)))

        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)     
            
        ## save
        file_path=args.output_dir+'RES_%s.txt'%args.model_name
        f1=open(file_path,'w')
        f1.write('Testing data: \n')
        f1.write('Picking error: \n')
        f1.write('MAE: %.2f (s) \n'%np.mean(abs(dt_p*0.01)))
        f1.close()
        ## save picking
        np.savez(args.output_dir+'new_test_p_erro',tp_tr=tp_tr,tp_t=tp_t)
        
        font2 = {'family': 'Times New Roman','weight': 'normal', 'size': 18,   }
        figure, ax = plt.subplots(figsize=(8,6))
        plt.hist(dt_p*0.01,40,edgecolor='black')
        plt.tick_params(labelsize=15)
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        plt.xlabel('tp_true-tp_pred (s)',font2)
        plt.ylabel('Frequency',font2)
        plt.savefig(args.output_dir+'P_error_new_testing.png',dpi=600)
        plt.show()
        
        # Recall Precision
        tp_up,tp_dn,tp_uw,ffp_up,ffp_dn,fp_up,fp_dn,fp_uw=confusion_matrix(pred2,test_label2,file_path,name='New testing')
        if args.save_result:
            np.savez(args.output_dir+'new_test_cm',res=np.array([tp_up,tp_dn,tp_uw,ffp_up,ffp_dn,fp_up,fp_dn,fp_uw],dtype=object))
        
        if args.plot_figure:
            if not os.path.exists(args.output_dir+'atten_map/'):
                os.mkdir(args.output_dir+'atten_map/')

            self_model1=Model(inputs=model.input, outputs=model.get_layer('Atten').output)
            save_path=True
            for na in ['tp_up','tp_dn','tp_uw','ffp_up','ffp_dn','fp_up','fp_dn','fp_uw']:
                if na=='tp_up':
                    cm=tp_up
                if na=='tp_dn':
                    cm=tp_dn
                if na=='tp_uw':
                    cm=tp_uw
                    
                if na=='ffp_up':
                    cm=ffp_up
                if na=='ffp_dn':
                    cm=ffp_dn
                    
                if na=='fp_up':
                    cm=fp_up  
                if na=='fp_dn':
                    cm=fp_dn                      
                if na=='fp_uw':
                    cm=fp_uw                      
                    
                for k in range(1,args.num_plots,2):
                    i=cm[k]
                    _,test_wt0=self_model1.predict(test_data[i:i+1,:,:], verbose=1)
                    wt=np.mean(test_wt0[0,:,:],axis=0).reshape(-1,1)
                    wt_map=np.repeat(wt,50,axis=1).T/np.max(wt)
                    wt_map=transform.resize(wt_map,(50,400))
                    #===========================#
                    fl=np.argmax(pred2[i])
                    if fl==0:
                        fl_p1='Up'
                    if fl==1:
                        fl_p1='Down'
                    if fl==2:
                        fl_p1='Unknown' 
                        
                    fl=np.argmax(test_label2[i])
                    if fl==0:
                        fl_tr='Up'
                    if fl==1:
                        fl_tr='Down'
                    if fl==2:
                        fl_tr='Unknown'
                    #===============================#
                    labelsize=22
                    font2 = {'family' : 'Times New Roman','weight' : 'bold','size' : 20,}
                    figure, axes = plt.subplots(3,1,figsize=(8,8))  
                    axes[0].plot(test_data[i,:,0],'k')
                    axes[0].set_xlim([0,400])
                    axes[0].set_ylim([-1,1])
                    axes[0].set_xticks(())
                    axes[0].tick_params(labelsize=labelsize)
                    axes[0].set_ylabel('Amplitude',font2)
                    axes[0].set_title('True: %s, Predicted: %s'%(fl_tr,fl_p1),font2)
                    labels = axes[0].get_xticklabels() + axes[0].get_yticklabels()
                    _=[label.set_fontname('Times New Roman') for label in labels]
                    
                    axes[1].plot(test_label1[i,:,0],'b',label='True')
                    axes[1].plot(pred1[i,:,0],'r-.',label='Predicted')
                    axes[1].set_xlim([0,400])
                    # axes[0].set_ylim([0,1])
                    axes[1].tick_params(labelsize=labelsize)
                    axes[1].set_title('The Probability of P arrival picking',font2)
                    labels = axes[1].get_xticklabels() + axes[1].get_yticklabels()
                    _=[label.set_fontname('Times New Roman') for label in labels]
                    axes[1].legend(prop=font2)
                    axes[1].set_ylabel('Probability',font2)
                    
                    axes[2].imshow( wt_map )
                    axes[2].set_xlim([0,400])
                    plt.xlabel('Samples',font2)
                    axes[2].tick_params(labelsize=labelsize)
                    axes[2].set_title('Attention Map',font2)
                    labels = axes[2].get_xticklabels() + axes[2].get_yticklabels()
                    _=[label.set_fontname('Times New Roman') for label in labels]
                    if save_path:
                        plt.savefig(args.output_dir+'atten_map/%s_%d.png'%(na,i),dpi=600)
                    plt.show()   
                
        ##SCSN ORG
        f5=h5py.File(args.org_dir,'r')
        ## get data label
        test_data2,test_label22=gen_test_data_UD(f5,2353054,classes=args.clas,time_shift=False) #2353054

        pred21,pred22=model.predict(test_data2)
        
        ## In[] QC time
        tp_t=np.argmax(pred21[:,:,0],axis=1)
        dt_p=200-tp_t
        print('MAE: %.2f (s)'%np.mean(abs(dt_p*0.01)))
        
        # save pciking error
        f1=open(file_path,'a+')
        f1.write('SCSN data: \n')
        f1.write('Picking error: \n')
        f1.write('MAE: %.2f (s) \n'%np.mean(abs(dt_p*0.01)))
        f1.close() 
        ## save picking
        if args.save_result:
            np.savez(args.output_dir+'scsn_test_p_erro',tp_t=tp_t)        
        
        font2 = {'family' : 'Times New Roman',
            'weight' : 'normal',
            'size'   : 18,
            }
        figure, ax = plt.subplots(figsize=(8,6))
        plt.hist(dt_p*0.01,40)
        plt.tick_params(labelsize=15)
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        plt.xlabel('tp_true-tp_pred (s)',font2)
        plt.ylabel('Frequency',font2)
        plt.savefig(args.output_dir+'scsn_time_error.png',dpi=600)
        plt.show()          
        ## confusion matrix
    
        tp_up2,tp_dn2,tp_uw2,ffp_up2,ffp_dn2,fp_up2,fp_dn2,fp_uw2=confusion_matrix(pred22,test_label22,file_path,name='SCSN test')
        if args.save_result:  
            np.savez(args.output_dir+'scsn_test_cm',res=np.array([tp_up2,tp_dn2,tp_uw2,ffp_up2,ffp_dn2,fp_up2,fp_dn2,fp_uw2],dtype=object))  
    
    elif args.mode=='predict':  
        print(args.mode)
        ## load model 
        try:
            model=load_model(args.model_dir+args.model_name+'.h5',custom_objects=SeqSelfAttention.get_custom_objects())
        except:
            print('Do not exists model!')  
            
        #============================#
        # example 1
        if args.exam=='example1':
            f5=h5py.File(args.org_dir,'r')
            test_data2,test_label22=gen_test_data_UD(f5,2353054,classes=args.clas,time_shift=False) #2353054
            pred_pick,pre_polatiry=model.predict(test_data2)   
        #exmaple 2
        if args.exam=='example2':
            f44=h5py.File(args.valid_dir,'r')
            validation_generator_test=DataGenerator_PP1_S_test(f44,len(f44['Y']),gaus,batch_size=args.batch_size, classes=args.clas)  
            pred_pick,pre_polatiry=model.predict_generator(validation_generator_test,verbose=1)
        if args.save_result:    
            np.savez(args.pre_result+'%s_%s'%(args.exam,args.model_name),pred_pick=pred_pick,pre_polatiry=pre_polatiry)
    
    else:
        print("mode should be: train, test, or predict")
        
# In[]

if __name__ == '__main__':
    args = read_args()
    start_gpu(args)
    if args.mode=='train':
        set_configure(args)
    main(args)
    print('Finish !!!')
    
    
    
    
    
