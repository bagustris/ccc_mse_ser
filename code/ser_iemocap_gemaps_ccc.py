# Dimensional speech emotion recognition
# To evaluate loss function (MSE vs CCC)
# Coded by Bagus Tris Atmaja (bagus@ep.its.ac.id)
# changelog
# 2020-02-13: Modified from gemaps-paa hfs
# 2020-02-14: Use 'tanh' activation to lock the output range in [-1, 1]
#             with RMSprop optimizer

import numpy as np
import pickle
import pandas as pd

import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense, CuDNNLSTM, Flatten, \
                         Embedding, Dropout, BatchNormalization, \
                         RNN, concatenate, Activation

from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

import random as rn
import tensorflow as tf

rn.seed(123)
np.random.seed(99)
tf.set_random_seed(1234)

# load feature and labels
feat = np.load('/home/s1820002/spro2020/data/feat_ws_3.npy')
vad = np.load('/home/s1820002/IEMOCAP-Emotion-Detection/y_egemaps.npy')

# use only mean and std
feat = feat[:,:-1]

# for LSTM input shape (batch, steps, features/channel)
#feat = feat.reshape(feat.shape[0], 1, feat.shape[1])

# remove outlier, < 1, > 5
vad = np.where(vad==5.5, 5.0, vad)
vad = np.where(vad==0.5, 1.0, vad)

# standardization
scaled_feature = True

# set Dropout
do = 0.3

if scaled_feature == True:
    scaler = StandardScaler()
    scaler = scaler.fit(feat) #.reshape(feat.shape[0]*feat.shape[1], feat.shape[2]))
    scaled_feat = scaler.transform(feat) #.reshape(feat.shape[0]*feat.shape[1], feat.shape[2]))
    #scaled_feat = scaled_feat.reshape(feat.shape[0], feat.shape[1], feat.shape[2])
    feat = scaled_feat
else:
    feat = feat

scaled_vad = True

# standardization
if scaled_vad:
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(vad) #.reshape(vad.shape[0]*vad.shape[1], vad.shape[2]))
    scaled_vad = scaler.transform(vad) #.reshape(vad.shape[0]*vad.shape[1], vad.shape[2]))
    vad = scaled_vad 
else:
    vad = vad

# Concordance correlation coefficient (CCC)-based loss function - using non-inductive statistics
def ccc(gold, pred):
    gold       = K.squeeze(gold, axis=-1)
    pred       = K.squeeze(pred, axis=-1)
    gold_mean  = K.mean(gold, axis=-1, keepdims=True)
    pred_mean  = K.mean(pred, axis=-1, keepdims=True)
    covariance = (gold-gold_mean)*(pred-pred_mean)
    gold_var   = K.mean(K.square(gold-gold_mean), axis=-1,  keepdims=True)
    pred_var   = K.mean(K.square(pred-pred_mean), axis=-1, keepdims=True)
    ccc        = K.constant(2.) * covariance / (gold_var + pred_var + K.square(gold_mean - pred_mean) + K.common.epsilon())
    return ccc


def ccc_loss(gold, pred):  
    # input (num_batches, seq_len, 1)
    ccc_loss   = K.constant(1.) - ccc(gold, pred)
    return ccc_loss

# reshape input feature for LSTM
feat = feat.reshape(feat.shape[0], 1, feat.shape[1])

# API model, if use RNN, first two rnn layer must return_sequences=True
def api_model(alpha, beta, gamma):
    # speech network
    input_speech = Input(shape=(feat.shape[1], feat.shape[2]), name='speech_input')
    net_speech = BatchNormalization()(input_speech)
    net_speech = CuDNNLSTM(feat.shape[2], return_sequences=True)(net_speech)
    net_speech = CuDNNLSTM(256, return_sequences=True)(net_speech)
    net_speech = CuDNNLSTM(256, return_sequences=False)(net_speech)
    #net_speech = Flatten()(net_speech)
    net_speech = Dense(64)(net_speech)
    #net_speech = Dropout(0.1)(net_speech)

    target_names = ('v', 'a', 'd')
    model_combined = [Dense(1, name=name, activation='tanh')(net_speech) for name in target_names]

    model = Model(input_speech, model_combined) 
    #model.compile(loss=ccc_loss, optimizer='rmsprop', metrics=[ccc])
    model.compile(loss=ccc_loss, 
                  loss_weights={'v': alpha, 'a': beta, 'd': gamma},
                  optimizer='rmsprop', metrics=[ccc, 'mse'])
    return model

#def main(alpha, beta, gamma):
model = api_model(0.1, 0.5, 0.4)
model.summary()

# 7869 first data of session 5 (for LOSO)
earlystop = EarlyStopping(monitor='val_loss', mode='min', patience=10,
                          restore_best_weights=True)
hist = model.fit(feat[:7869], vad[:7869].T.tolist(), batch_size=64, #best:8
                  validation_split=0.2, epochs=200, verbose=1, shuffle=True, 
                  callbacks=[earlystop])
metrik = model.evaluate(feat[7869:], vad[7869:].T.tolist())
print('CCC= ', np.array(metrik)[[-6,-4,-2]])
print('MSE= ', np.array(metrik)[[-5,-3,-1]])

# Plot scatter
#va = vad[7869:, :-1]
#predik_vad = model.predict(feat[7869:], batch_size=64)
#predik_va =  np.array(predik_vad).T.reshape(2170,3)[:,:-1]
#import matplotlib.pyplot as plt
#plt.scatter(va[:,0], va[:,1])
#plt.scatter(predik_va[:,0], predik_va[:,1])
#plt.savefig('scatter_gemaps_mse.pdf')
## check max min
#predik_va.max()
#predik_va.min()
