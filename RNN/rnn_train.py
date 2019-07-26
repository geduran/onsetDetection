# main.py
# RNN of 32x32 patches (grayvalues) of two classes (0 1nd 1)
# (c) D.Mery, 2019

import sys
import numpy         as     np
from   keras         import backend as K
import tensorflow    as     tf
from   keras.backend import tensorflow_backend
from   rnn_utils     import sequentialRNN, defineCallBacks
from   rnn_utils     import loadAudioPatches, Metrics


# input arguments
# 1 > type of execution: 0:training, 1: eval testing only, 2: eval training and testing, 3: layer's output
# 2 > 1: output layer of train/testing data is stored in train_output.npy/testing_output.npy, 0 = no
# 3 > Complexity of data. 1: Only Bass. 2: Bass+Percussive. 3: Bass+Percussive+Chords
# Example: Python3 main.py eyenose.mat 0 # for training and testing
# Example: Python3 main.py eyenose.mat 1 # for testing only (after training)
if len(sys.argv) < 3:
    raise Exception('Arguments should be 3, but got {}'.format(len(sys.argv))+
                    '\nArgument 1: Type of onset, Argument 2: Type of DB')

onsetType = sys.argv[1]
stage = sys.argv[2]

epochs   = 100 # maximal number of epochs in training stage
n_hidden = 40

batch_size    = 128

# tensorflow configuration
K.set_image_dim_ordering('th')
print(K.image_data_format())
config  = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
tensorflow_backend.set_session(session)


audioPath = '../MIDI/Train/'+str(stage)+'/rnn'+onsetType+'Data_cqt_mel.pkl'

best_model    = 'best_model_'+str(stage)+'_rnn'+onsetType+'.h5'
last_model    = 'last_model_'+str(stage)+'_rnn'+onsetType+'.h5'
# prepare callbacks
callbacks    = defineCallBacks(best_model)

X_train, Y_train, X_test, Y_test, classes = loadAudioPatches(audioPath)#loadPatches(patches_file)

num_classes  = len(classes)
num_samples = X_train.shape[0]
input_shape  = (None, X_test.shape[2])#(X_test.shape[1], X_test.shape[2])
class_proportion = np.sum(Y_train[:,:,0])/np.sum(Y_train[:,:,1])

# RNN architecture defintion
model = sequentialRNN(input_shape,num_classes,n_hidden)

print('class_proportion: {}'.format(class_proportion))
history = model.fit(X_train, Y_train,
      batch_size      = batch_size,
      epochs          = epochs,
      verbose         = 1,
      validation_data = (X_test, Y_test),
      # class_weight    = {0: 1., 1: class_proportion},
      shuffle         = True,
      callbacks       = callbacks)
model.save_weights(last_model)
