# rnn_utils.py (see example in main.py)
# RNN of 32x32 patches (grayvalues) of two classes (0 1nd 1)
# (c) D.Mery, 2019

import numpy                as np
import matplotlib.pyplot    as plt
import keras
from   scipy.io             import loadmat
from   keras.models         import Sequential, Model
from keras.layers           import GRU, LSTM, SimpleRNN, Dropout, Dense, Activation, Bidirectional
#from   keras.layers         import Flatten, BatchNormalization, regularizers
#from   keras.layers         import Conv2D, MaxPooling2D
from   keras.callbacks      import ModelCheckpoint, EarlyStopping, Callback
from   keras.utils.np_utils import to_categorical
import keras.backend as K
from sklearn.metrics import confusion_matrix, precision_score, f1_score, recall_score
import os
import tensorflow as tf
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

    RNN_type = LSTM

    #Start Neural Network
    model = Sequential()

    # model.add(Bidirectional(RNN_type(n_hidden, return_sequences=True),
                            # input_shape=input_shape))

    # model.add(Bidirectional(RNN_type(n_hidden, return_sequences=False)))
    model.add(RNN_type(n_hidden, input_shape=input_shape,
                                   return_sequences=True))

    model.add(RNN_type(n_hidden, return_sequences=False))

    # model.add(RNN_type(n_hidden, return_sequences=False))
    model.add(Dropout(0.25))

    #Fully connected final layer
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.compile(loss        = keras.losses.binary_crossentropy,
                  optimizer   = keras.optimizers.RMSprop(),
                  metrics     = ['accuracy', f1, recall, precision])

    model.summary()

    return model


def deleteWeights(best_model,last_model):
    if os.path.exists(best_model):
        os.remove(best_model)
    if os.path.exists(last_model):
        os.remove(last_model)

def evaluateRNN(model,X,y,st):
    print('evaluating performance in '+st+' set ('+str(y.shape[0])+' samples)...')
    score   = model.evaluate(X,y,verbose=0)
    print(st+' loss:', score[0])
    print(st+' accuracy:', score[1])

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
    labels, mel1, mel2, mel3 = pickle.load(file)
    file.close()

    mx_mel1 = np.max(mel1)
    mn_mel1 = np.min(mel1)
    mel1 = (mel1-mn_mel1) / (mx_mel1-mn_mel1)

    mx_mel2 = np.max(mel2)
    mn_mel2 = np.min(mel2)
    mel2 = (mel2-mn_mel2) / (mx_mel2-mn_mel2)

    mx_mel3 = np.max(mel3)
    mn_mel3 = np.min(mel3)
    mel3 = (mel3-mn_mel3) / (mx_mel3-mn_mel3)

    seq_len = 10

    print('mel1.shape {}, mel2.shape {}, mel3.shape {}'.format(mel1.shape, mel2.shape, mel3.shape))
    # samples = np.concatenate((mel1, mel2, mel3), axis=1)
    samples = mel1

    print('samples.shape {}'.format(samples.shape))

    n_samples = samples.shape[0] - seq_len
    n_features = samples.shape[1]
    ind = set()
    while len(ind) < n_samples:
        ind.add(random.randint(0, samples.shape[0]-seq_len-1))

    X_train = np.zeros((n_samples, seq_len, n_features))
    Y_train = np.zeros((n_samples, 2))

    cont = 0
    for i in ind:
        print('Cargando Datos {}/{}'.format(cont,n_samples), end='\r')
        X_train[cont,:,:] = samples[i:i+seq_len,:]
        if np.sum(labels[i+2:i+seq_len-2]):
            Y_train[cont,1] = 1
        else:
            Y_train[cont,0] = 1
        cont += 1

    n_test = n_samples//20
    testIndex = set()
    while len(testIndex) < n_test:
        testIndex.add(random.randint(0, n_samples-n_test))

    testIndex = list(testIndex)

    X_test = X_train[testIndex,:,:]
    Y_test = Y_train[testIndex,:]
    X_train = np.delete(X_train, testIndex,axis=0)
    Y_train = np.delete(Y_train, testIndex,axis=0)
    print('x_train: {}, x_test: {}, y_train: {}, y_test: {}'.format(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape))

    classes  = [0, 1]
    return X_train, Y_train, X_test, Y_test, classes


def evaluateLayer(model,K,X,st,num_layer,test_predict):
    inp        = model.input                                           # input placeholder
    outputs    = [layer.output for layer in model.layers]              # all layer outputs
    functor    = K.function([inp, K.learning_phase()], outputs )       # evaluation function

    test       = X[0]
    test       = test.reshape(1,1,X.shape[2],X.shape[3])
    layer_outs = functor([test, 1.])
    x          = layer_outs[num_layer]
    n          = X.shape[0]
    m          = x.shape[1]

    if test_predict == 1:
        print('computing prediction output in ' +st +' set with '+str(n)+' samples...')
        y = model.predict(X)
        print('saving prediction in '+st+'_predict.npy ...')
        np.save(st+'_predict',y)

    if num_layer>0:
        d = np.zeros((n,m))
        print('computing output layer '+str(num_layer)+ ' in ' +st +' set with '+str(n)+' descriptors of '+str(m)+' elements...')
        for i in range(n):
            test       = X[i]
            test       = test.reshape(1,1,X.shape[2],X.shape[3])
            layer_outs = functor([test, 1.])
            d[i]       = layer_outs[num_layer]
        print('saving layer output in '+st+'_layer_'+str(num_layer)+'.npy ...')
        np.save(st+'_layer_'+str(num_layer),d)

def plotCurves(history):
    # loss curves
    print('displaying loss and accuracy curves...')
    plt.figure(figsize=[8,6])
    plt.plot(history.history['loss'],'r',linewidth=3.0)
    plt.plot(history.history['val_loss'],'b',linewidth=3.0)
    plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Loss',fontsize=16)
    plt.title('Loss Curves',fontsize=16)
    plt.show()

    # accuracy curves
    plt.figure(figsize=[8,6])
    plt.plot(history.history['acc'],'r',linewidth=3.0)
    plt.plot(history.history['val_acc'],'b',linewidth=3.0)
    plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Accuracy',fontsize=16)
    plt.title('Accuracy Curves',fontsize=16)
    plt.show()

def computeConfussionMatrix(model,X,y):
    print('computing confussion matrix...')
    Y_prediction = model.predict(X)
    Y_pred_classes = np.argmax(Y_prediction,axis = 1) # classes to one hot vectors
    Y_true = np.argmax(y,axis = 1)                    # classes to one hot vectors
    confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)
    print(confusion_mtx)
    plt.figure(figsize=(10,8))
    heatmap(confusion_mtx, annot=True, fmt="d")
    plt.show()
