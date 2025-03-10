# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 10:05:46 2021

@author: ZeZhou Hou
"""
import os
NUM=0#
os.environ["CUDA_VISIBLE_DEVICES"] = str(NUM)#

import tensorflow as tf
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth=True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)#

import Resnet_speck as train_net

import numpy as np
import os
from time import time

def mkdir(Path):
    folder=os.path.exists(Path)
    if not folder:
        os.makedirs(Path)
        print('No Folder')
        
Parameter=np.load('parameter.npy')

t0=time()

x=1#Hamming weight
ROUND=7


for par in Parameter:
    pi=par.tolist()
    mkdir('./hw='+str(x)+'/'+str(pi))
    a=np.load('SPECK32_Round='+str(ROUND)+'hw='+str(x)+'.npy')
    PATH='./hw='+str(x)+'/'+str(pi)+'/'
    Name=os.listdir(PATH)

    for j in range(len(a)):
        print(pi,str(j)+'/'+str(len(a)))
        if str(a[j])+str(ROUND)+'.txt' not in Name:
            print(a[j])
            train_net.train_model(10**7,ROUND,a[j],pi,PATH)

    

    
