# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 20:31:06 2020

@author: ZeZhou Hou
"""

import speck as ciph
import pickle

import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv1D, Input, Reshape, Permute, Add, Flatten, BatchNormalization, Activation
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.regularizers import l2



def create_model():
    num_blocks=int(ciph.block_size()/ciph.WORD_SIZE())
    num_filters=ciph.block_size()
    num_outputs=1
    d1=64
    d2=64
    word_size=ciph.WORD_SIZE()
    ks=3
    reg_param=0.0001
    inp = Input(shape=(num_blocks * word_size * 2,));
    rs = Reshape((2 * num_blocks, word_size))(inp);
    perm = Permute((2,1))(rs);
    #add a single residual layer that will expand the data to num_filters channels
    #this is a bit-sliced layer
    conv0 = Conv1D(num_filters, kernel_size=1, padding='same', kernel_regularizer=l2(reg_param))(perm);
    conv0 = BatchNormalization()(conv0);
    conv0 = Activation('relu')(conv0);
    #add residual blocks
    shortcut = conv0;
    for i in range(5):
        conv1 = Conv1D(num_filters, kernel_size=ks, padding='same', kernel_regularizer=l2(reg_param))(shortcut);
        conv1 = BatchNormalization()(conv1);
        conv1 = Activation('relu')(conv1);
        conv2 = Conv1D(num_filters, kernel_size=ks, padding='same',kernel_regularizer=l2(reg_param))(conv1);
        conv2 = BatchNormalization()(conv2);
        conv2 = Activation('relu')(conv2);
        shortcut = Add()([shortcut, conv2]);
    #add prediction head
    flat1 = Flatten()(shortcut);
    dense1 = Dense(d1,kernel_regularizer=l2(reg_param))(flat1);
    dense1 = BatchNormalization()(dense1);
    dense1 = Activation('relu')(dense1);
    dense2 = Dense(d2, kernel_regularizer=l2(reg_param))(dense1);
    dense2 = BatchNormalization()(dense2);
    dense2 = Activation('relu')(dense2);
    out = Dense(num_outputs, activation='sigmoid', kernel_regularizer=l2(reg_param))(dense2);
    model = Model(inputs=inp, outputs=out);
    return(model)

def train_model(num,Round,diff,pi,PATH):
    net_name=str(diff)
    x_round=Round
    data_x,data_y=ciph.make_train_data(num,x_round,diff,pi)
    data_x=data_x.astype(np.uint8)
    data_y=data_y.astype(np.uint8)

    x_val,y_val=ciph.make_train_data(int(num/100),x_round,diff,pi)
    x_val=x_val.astype(np.uint8)
    y_val=y_val.astype(np.uint8)


    seed=199847
    np.random.seed(seed)
    model=create_model()
    model.compile(optimizer='adam',loss='mse',metrics=['acc'])
    filepath_net=PATH+net_name+'weight'+str(Round)+'.h5'
    checkpoint=ModelCheckpoint(filepath=filepath_net,monitor='val_acc',verbose=1,save_best_only=True,mode='max')
    callback_list=[checkpoint]
    history=model.fit(data_x,data_y,validation_data=(x_val,y_val),epochs=15,batch_size=5000,verbose=1,callbacks=callback_list)
    with open(PATH+net_name+str(Round)+'.txt','wb') as file:        
        pickle.dump(history.history,file)
    model_json=model.to_json()
    with open(PATH+net_name+'model'+'.json','w') as file:
        file.write(model_json)

