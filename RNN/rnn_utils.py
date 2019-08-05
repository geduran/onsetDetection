# rnn_utils.py (see example in main.py)
# RNN of 32x32 patches (grayvalues) of two classes (0 1nd 1)
# (c) D.Mery, 2019

import numpy                as np
import matplotlib.pyplot    as plt
import keras
from   scipy.io             import loadmat
from   keras.models         import Sequential, Model
from   keras.layers         import GRU, LSTM, SimpleRNN, Dropout, Dense
from   keras.layers         import Activation, Bidirectional, TimeDistributed
#from   keras.layers         import Flatten, BatchNormalization, regularizers
#from   keras.layers         import Conv2D, MaxPooling2D
from   keras.callbacks      import ModelCheckpoint, EarlyStopping, Callback
from   keras.utils.np_utils import to_categorical
import keras.backend        as K
from sklearn.metrics        import confusion_matrix, precision_score, f1_score, recall_score
import sklearn.model_selection
import os
import tensorflow           as tf
import random
import pickle

DATA_PATH = ''


def focal_loss(y_true, y_pred):
    gamma = 2.0
    alpha = 0.25
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_pred[:,1], 0, 1)) * K.round(K.clip(y_true[:,1], 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true[:,1], 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_pred[:,1], 0, 1)) * K.round(K.clip(y_true[:,1], 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred[:,1], 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_pred[:,1], 0, 1)) * K.round(K.clip(y_true[:,1], 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true[:,1], 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_pred[:,1], 0, 1)) * K.round(K.clip(y_true[:,1], 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred[:,1], 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict_ = (np.asarray(self.model.predict(self.validation_data[0])))
        #print('val_predict: {}'.format(val_predict_))
        val_targ_ = self.validation_data[1]
        val_targ = []
        for tr in val_targ_:
            val_targ.append(np.argmax(tr))

        val_predict = []
        for pr in val_predict_:
            val_predict.append(np.argmax(pr))

        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        _val_f1 = f1_score(val_targ, val_predict)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print('— val_f1: %f — val_precision: %.2f — val_recall %.2f'%(_val_f1,
                                                  _val_precision, _val_recall))

def sequentialRNN(input_shape,num_classes,n_hidden):

    RNN_type = GRU

    #Start Neural Network
    model = Sequential()

    model.add(Bidirectional(RNN_type(n_hidden, return_sequences=True),
                            input_shape=input_shape))

    model.add(Bidirectional(RNN_type(n_hidden, return_sequences=True)))

    model.add(Bidirectional(RNN_type(n_hidden, return_sequences=True)))

    # model.add(RNN_type(n_hidden, return_sequences=True,
    #           input_shape=input_shape))
    # model.add(RNN_type(n_hidden, return_sequences=True))
    # model.add(Dropout(0.5))


    model.add(TimeDistributed(Dense(num_classes, activation='softmax')))


    model.compile(loss        = [focal_loss],#keras.losses.binary_crossentropy,
                  optimizer   = keras.optimizers.RMSprop(),
                  metrics     = ['accuracy', f1, recall, precision])

    model.summary()

    return model

def defineCallBacks(model_file):
    #metrics = Metrics()
    callbacks = [
        EarlyStopping(
            monitor        = 'val_f1',#'val_acc',
            patience       = 10,
            mode           = 'max',
            verbose        = 1),
        ModelCheckpoint(model_file,
            monitor        = 'val_f1',#'val_acc',
            save_best_only = True,
            mode           = 'max',
            verbose        = 0)#,
        #metrics
    ]
    return callbacks


def loadAudioPatches(st_file):
    file = open(st_file, 'rb')
    _labels, mel1, mel2, mel3 = pickle.load(file)
    file.close()

    labels = np.zeros(_labels.shape)
    labels[1:] = _labels[:-1]
    # labels = _labels

    mel1 = (mel1 - mel1.mean(axis=0)) / mel1.std(axis=0)
    mel2 = (mel2 - mel2.mean(axis=0)) / mel2.std(axis=0)
    mel3 = (mel3 - mel3.mean(axis=0)) / mel3.std(axis=0)

    # diff1 = np.zeros(mel1.shape)
    # diff1[1:,:] = mel1[1:,:] - mel1[:-1,:]
    # diff2 = np.zeros(mel2.shape)
    # diff2[1:,:] = mel2[1:,:] - mel2[:-1,:]
    # diff3 = np.zeros(mel3.shape)
    # diff3[1:,:] = mel3[1:,:] - mel3[:-1,:]

    seq_len = 400

    # print('mel1.shape {}, mel2.shape {}, mel3.shape {}'.format(mel1.shape, mel2.shape, mel3.shape))
    samples = np.concatenate((mel1, mel2, mel3), axis=1)

    print('samples.shape {}'.format(samples.shape))

    ind_ = np.arange(0, samples.shape[0] - seq_len, int(seq_len/8))
    n_samples = len(ind_)

    ind = []
    for i in ind_:
        if not np.sum(labels[i:i+10]):
            ind.append(i)
        else:
            ind.append(i-12)

    n_features = samples.shape[1]
    # ind = set()
    # while len(ind) < n_samples:
    #     ind.add(random.randint(0, samples.shape[0]-seq_len-1))

    X_train = np.zeros((n_samples, seq_len, n_features))
    Y_train = np.zeros((n_samples, seq_len, 2))

    cont = 0
    for i in ind:
        print('Cargando Datos {}/{}'.format(cont,n_samples), end='\r')
        X_train[cont,:,:] = samples[i:i+seq_len,:]
        Y_train[cont,:,:] = to_categorical(labels[i:i+seq_len])
        cont += 1


    split_data = sklearn.model_selection.train_test_split(X_train, Y_train,
                                                          test_size=0.05)
    X_train, X_test, Y_train, Y_test = split_data
    print('x_train: {}, x_test: {}, y_train: {}, y_test: {}'.format(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape))

    classes  = [0, 1]
    return X_train, Y_train, X_test, Y_test, classes
