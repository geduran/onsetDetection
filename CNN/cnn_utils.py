import numpy                as np
import matplotlib.pyplot    as plt
import keras
from   scipy.io             import loadmat
from   keras.models         import Sequential, Model
from   keras.layers         import Dense, Conv2D, Flatten, Activation, Dropout
from   keras.layers         import regularizers, MaxPooling2D, BatchNormalization
from   keras.callbacks      import ModelCheckpoint, EarlyStopping, Callback
from   keras.utils.np_utils import to_categorical
import keras.backend as K
import tensorflow as tf
from sklearn.metrics import confusion_matrix, precision_score, f1_score, recall_score
import sklearn.model_selection
import os
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

def sequentialCNN(input_shape):


    #Start Neural Network
    model = Sequential()

    #convolution 1st layer
    model.add(Conv2D(64, kernel_size=(5, 5), padding="same",activation="relu",
                     input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())
    # model.add(Dropout(droprate))

    #convolution layer i
    model.add(Conv2D(32, kernel_size=(3,3), activation="relu",padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())
    # model.add(Dropout(droprate))

    #Fully connected layers
    model.add(Flatten())
    model.add(Dense(200 , use_bias=True))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    #Fully connected final layer
    model.add(Dense(2))
    model.add(Activation('softmax'))

    model.compile(loss        = keras.losses.binary_crossentropy,
                  optimizer   = keras.optimizers.Adam(),
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
    labels[3:] = _labels[:-3]

    mel1 = (mel1 - mel1.mean(axis=0)) / mel1.std(axis=0)
    mel2 = (mel2 - mel2.mean(axis=0)) / mel2.std(axis=0)
    mel3 = (mel3 - mel3.mean(axis=0)) / mel3.std(axis=0)

    seq_len = 10

    samples = mel1

    print('samples.shape {}'.format(samples.shape))

    n_samples = samples.shape[0] - seq_len
    n_features = samples.shape[1]

    ind = set()
    while len(ind) < n_samples:
        ind.add(random.randint(0, samples.shape[0]-seq_len-1))

    X_train = np.zeros((n_samples, 3, seq_len, n_features))
    Y_train = np.zeros((n_samples, 2))

    cont = 0
    for i in ind:
        print('Cargando Datos {}/{}'.format(cont,n_samples), end='\r')
        X_train[cont,0,:,:] = mel1[i:i+seq_len,:]
        X_train[cont,1,:,:] = mel2[i:i+seq_len,:]
        X_train[cont,2,:,:] = mel3[i:i+seq_len,:]
        if np.sum(labels[i+4:i+6]):
            Y_train[cont,1] = 1
        else:
            Y_train[cont,0] = 1
        cont += 1

    split_data = sklearn.model_selection.train_test_split(X_train, Y_train,
                                                          test_size=0.05)
    X_train, X_test, Y_train, Y_test = split_data
    print('x_train: {}, x_test: {}, y_train: {}, y_test: {}'.format(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape))

    classes  = [0, 1]
    return X_train, Y_train, X_test, Y_test, classes
