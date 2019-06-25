# main.py
# CNN of 32x32 patches (grayvalues) of two classes (0 1nd 1)
# (c) D.Mery, 2019

import sys
import numpy         as     np
from   keras         import backend as K
import tensorflow    as     tf
from   keras.backend import tensorflow_backend
from   cnn_utils     import sequentialCNN, evaluateCNN, defineCallBacks, loadPatches, loadTestPatches, loadAudioPatches
from   cnn_utils     import evaluateLayer, plotCurves, computeConfussionMatrix, deleteWeights, Metrics


# input arguments
# 1 > type of execution: 0:training, 1: eval testing only, 2: eval training and testing, 3: layer's output
# 2 > 1: output layer of train/testing data is stored in train_output.npy/testing_output.npy, 0 = no
# 3 > Complexity of data. 1: Only Bass. 2: Bass+Percussive. 3: Bass+Percussive+Chords
# Example: Python3 main.py eyenose.mat 0 # for training and testing
# Example: Python3 main.py eyenose.mat 1 # for testing only (after training)

if len(sys.argv) < 3:
    raise Exception('Arguments should be 3, but got {}'.format(len(sys.argv))+
                    '\nArgument 1: Type of onset, Argument 2: Type of DB')


# definition of the CNN architecture:
# there are  n = len(p)  conv2D layers, and m = len(f) fully connected layers
# each conv2D layer has a mask of size p[i] x p[i] x d[i], i=0...n-1
# each conv2D layer has: convolution, ReLU, MaxPooling and Dropout (see details in sequentialCNN in cnn_utils.py)
# each fully connected layer as f[j] nodes, j = 0...m-1

# parameters
model_file    = 1  # 1 best trained model, 0 last trained model
epochs        = 100 # maximal number of epochs in training stage


type_exec    = 0#int(sys.argv[1]) # 0:training, 1: eval testing only, 2: eval training and testing, 3: layer's output


if     type_exec == 0: # training
    print('Execution Type 0: Training...')
    do_train      = 1  # number of epochs, 0 means no train, only evaluation
    only_test     = 0  # 1 loads only test data (eg. for sliding windows)
    ev_train      = 1  # accuracy on train data is computed
    ev_test       = 1  # accuracy on test data is computed
    do_conf_mtx   = 1  # compute confussion matrix on testing data
    train_predict = 0  # prediction output of train data is stored in train_predic.npy, 0 = no
    test_predict  = 0  # prediction output of test data is stored in test_predic.npy, 0 = no
    train_layer   = 0  # output layer of train data is stored in train_output.npy, 0 = no
    test_layer    = 0  # output layer of test data is stored in test_output.npy, 0 = no

elif   type_exec == 1: # eval on testing only
    print('Execution Type 1: Evaluation on testing set only...')
    do_train      = 0  # number of epochs, 0 means no train, only evaluation
    only_test     = 0  # 1 loads only test data (eg. for sliding windows)
    ev_train      = 0  # accuracy on train data is computed
    ev_test       = 1  # accuracy on test data is computed
    do_conf_mtx   = 1  # compute confussion matrix on testing data
    train_predict = 0  # prediction output of train data is stored in train_predic.npy, 0 = no
    test_predict  = 0  # prediction output of test data is stored in test_predic.npy, 0 = no
    train_layer   = 0  # output layer of train data is stored in train_output.npy, 0 = no
    test_layer    = 0  # output layer of test data is stored in test_output.npy, 0 = no

elif   type_exec == 2: # eval on training and testing
    print('Execution Type 2: Evaluation on training and testing set...')
    do_train      = 0  # number of epochs, 0 means no train, only evaluation
    only_test     = 0  # 1 loads only test data (eg. for sliding windows)
    ev_train      = 1  # accuracy on train data is computed
    ev_test       = 1  # accuracy on test data is computed
    do_conf_mtx   = 1  # compute confussion matrix on testing data
    train_predict = 0  # prediction output of train data is stored in train_predic.npy, 0 = no
    test_predict  = 0  # prediction output of test data is stored in test_predic.npy, 0 = no
    train_layer   = 0  # output layer of train data is stored in train_output.npy, 0 = no
    test_layer    = 0  # output layer of test data is stored in test_output.npy, 0 = no

elif   type_exec == 3: # eval layer's output training and testing
    print('Execution Type 3: Computation of layer-output...')
    do_train      = 0  # number of epochs, 0 means no train, only evaluation
    only_test     = 0  # 1 loads only test data (eg. for sliding windows)
    ev_train      = 0  # accuracy on train data is computed
    ev_test       = 0  # accuracy on test data is computed
    do_conf_mtx   = 0  # compute confussion matrix on testing data
    train_predict = 0  # prediction output of train data is stored in train_predic.npy, 0 = no
    test_predict  = 0  # prediction output of test data is stored in test_predic.npy, 0 = no
    train_layer   = 0 #sys.argv[2]  # output layer of train data is stored in train_output.npy, 0 = no
    test_layer    = 0 #sys.argv[2]  # output layer of test data is stored in test_output.npy, 0 = no


#size of parameters
batch_size    = 128
droprate      = 0.25

# tensorflow configuration
K.set_image_dim_ordering('th')
print(K.image_data_format())
config  = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
tensorflow_backend.set_session(session)


onsetType = sys.argv[1]
stage = sys.argv[2]
audioPath = '../MIDI/Train/'+str(stage)+'/cnn'+onsetType+'Data_cqt_mel.pkl'

plot_curves   = 1  # plot loss and accuracy curves
best_model    = 'best_model_'+str(stage)+'_cnn'+onsetType+'.h5'
last_model    = 'last_model_'+str(stage)+'_cnn'+onsetType+'.h5'
# prepare callbacks
callbacks    = defineCallBacks(best_model)

# load patches
if only_test == 1:
    X_test, Y_test, classes = loadAudioPatches(audioPath)
else:
    X_train, Y_train, X_test, Y_test, classes = loadAudioPatches(audioPath)#loadPatches(patches_file)

num_classes  = len(classes)
num_samples = X_train.shape[0]
input_shape  = (X_test.shape[1], X_test.shape[2],X_test.shape[3])
class_proportion = np.sum(Y_train[:,0])/np.sum(Y_train[:,1])

# CNN architecture defintion
model = sequentialCNN(input_shape,num_classes)

# training
if do_train == 1 and only_test == 0:
    #deleteWeights(best_model,last_model)
    history = model.fit(X_train, Y_train,
          batch_size      = batch_size,
          epochs          = epochs,
          verbose         = 1,
          validation_data = (X_test, Y_test),
          class_weight    = {0: 1., 1: class_proportion},
          shuffle         = True,
          callbacks       = callbacks)
    model.save_weights(last_model)
    if plot_curves == 1:
        plotCurves(history)

# load trained model
if model_file == 1:
    print('loading best model from '+best_model+' ...')
    model.load_weights(best_model)
else:
    print('loading last model from ' +last_model+' ...')
    model.load_weights(last_model)

# print results in training/testing data
print('results using audio:')
if ev_train == 1:
    evaluateCNN(model,X_train,Y_train,'training')
if ev_test == 1:
    evaluateCNN(model,X_test,Y_test,'testing')

# confussion matrix
if do_conf_mtx == 1:
    computeConfussionMatrix(model,X_test,Y_test)

# evaluation of layer outputs
if test_layer > 0  or test_predict == 1 :
    evaluateLayer(model,K,X_test,'test',test_layer,test_predict)
if train_layer > 0 or train_predict == 1:
    evaluateLayer(model,K,X_train,'train',train_layer,train_predict)
