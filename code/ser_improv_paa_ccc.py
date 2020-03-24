# ser_improv_paa_ccc.py 
# speech emotion recognition for MSP-IMPROV dataset with pyAudioAnalysis
# HFS features using CCC-based loss function
# coded by Bagus Tris Atmaja (bagus@ep.its.ac.id)
# changelog:
# 2020-02-13: Inital code, modified from deepMLP repo

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
# loading file and label
feat_train = np.load('/home/s1820002/ccc_mse/data/feat_hfs_paa_msp_train.npy')
feat_test = np.load('/home/s1820002/ccc_mse/data/feat_hfs_paa_msp_test.npy')

feat = np.vstack([feat_train, feat_test])

list_path = '/home/s1820002/msp-improv/helper/improv_data.csv'
list_file = pd.read_csv(list_path, index_col=None)
list_file = pd.DataFrame(list_file)
data = list_file.sort_values(by=['wavfile'])

vad_train = []
vad_test = []

for index, row in data.iterrows(): 
    #print(row['wavfile'], row['v'], row['a'], row['d']) 
    if int(row['wavfile'][18]) in range(1,6): 
        #print("Process vad..", row['wavfile']) 
        vad_train.append([row['v'], row['a'], row['d']]) 
    else: 
        #print("Process..", row['wavfile']) 
        vad_test.append([row['v'], row['a'], row['d']])

vad = np.vstack([vad_train, vad_test])

# standardization
scaled_feature = False

if scaled_feature:
    scaler = StandardScaler()
    scaler = scaler.fit(feat)
    scaled_feat = scaler.transform(feat)
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

# reshape feat size to match LSTM config
feat = feat.reshape(feat.shape[0], 1, feat.shape[1])

# train/test split, LOSO
X_train = feat[:len(feat_train)]
X_test = feat[len(feat_train):]
y_train = vad[:len(vad_train)]
y_test = vad[len(vad_train):]


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


# API model, if use RNN, first two rnn layer must return_sequences=True
def api_model():
    inputs = Input(shape=(feat.shape[1], feat.shape[2]), name='feat_input')
    net = BatchNormalization()(inputs)
    #net = Bidirectional(LSTM(64, return_sequences=True, dropout=do, recurrent_dropout=do))(net)
    net = CuDNNLSTM(feat.shape[2], return_sequences=True)(net)
    net = CuDNNLSTM(256, return_sequences=True)(net)
    net = CuDNNLSTM(256, return_sequences=False)(net)
    net = Dense(64)(net)
    #net = Dropout(0.1)(net)
    target_names = ('v', 'a', 'd')
    outputs = [Dense(1, name=name, activation='tanh')(net) for name in target_names]

    model = Model(inputs=inputs, outputs=outputs) #=[out1, out2, out3])
    model.compile(loss=ccc_loss, #{'v': ccc_loss, 'a': ccc_loss, 'd': ccc_loss}, 
                  loss_weights={'v': 0.3, 'a': 0.6, 'd': 0.1},
                  optimizer='rmsprop', metrics=[ccc, 'mse'])
    return model


model2 = api_model()
model2.summary()

earlystop = EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True)
hist = model2.fit(X_train, y_train.T.tolist(), batch_size=64,
                  validation_split=0.2, epochs=50, verbose=1, shuffle=True,
                  callbacks=[earlystop])

metrik = model2.evaluate(X_test, y_test.T.tolist())

print('CCC= ', np.array(metrik)[[-6,-4,-2]])
print('MSE= ', np.array(metrik)[[-5,-3,-1]])
