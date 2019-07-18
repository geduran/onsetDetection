
import glob
import warnings
import sys
sys.path.insert(0, '/home/geduran/Environments/onsetDetection/CNN')
sys.path.insert(0, '/home/geduran/Environments/onsetDetection/RNN')
import pandas        as pd
import numpy         as     np
import tensorflow    as     tf
from audioData       import *
from   keras         import backend as K
from   keras.backend import tensorflow_backend
from   cnn_utils     import sequentialCNN, defineCallBacks, loadAudioPatches
from   cnn_utils     import computeConfussionMatrix, deleteWeights
from   rnn_utils     import sequentialRNN, defineCallBacks, loadAudioPatches
from   rnn_utils     import computeConfussionMatrix, deleteWeights, Metrics
from midiData        import *
from dataManager     import *

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.filterwarnings(action="ignore", category=DeprecationWarning)


# Argv[1] type of DB to be analyzed

results = {}

HPSS = False

K.set_image_dim_ordering('th')
print(K.image_data_format())
config  = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
tensorflow_backend.set_session(session)

input_shape  = (2, 32, 24)
batch_size    = 128
droprate      = 0.25
num_classes  = 2
n_hidden = 40
n_layers = 1


if len(sys.argv) >= 2:
    listItems = [sys.argv[1]]
else:
    listItems = ['1', '2', '3', 'all']

if len(sys.argv) >= 3:
    listDB = [sys.argv[2]]
else:
    listDB = ['Train', 'Validation']


for trainVal in listDB:
    for mode in listItems:
        print('Analizando mode {}'.format(mode))
        midi_files = glob.glob('/home/geduran/Environments/onsetDetection/MIDI/' +
                                trainVal+'/'+mode+'/*.mid')

        num_files = len(midi_files)

        cnn_model_path = ('/home/geduran/Environments/onsetDetection/CNN/best_model_' +
                          mode+'_cnnBass.h5')

        cnn_model = sequentialCNN(input_shape,num_classes)

        cnn_model.load_weights(cnn_model_path)

        input_shape  = (10, 98)
        rnn_model_path = ('/home/geduran/Environments/onsetDetection/RNN/best_model_' +
                          mode+'_rnnBass.h5')
        rnn_model = sequentialRNN(input_shape,num_classes,n_hidden)
        rnn_model.load_weights(rnn_model_path)

        BM = BassManager()

        #for midi_path in midi_files:
        for i in range(num_files):
            if i > len(midi_files)-1:
                break
            midi_path = midi_files[i]
            midi = MidiData(midi_path)
            audio = AudioData(midi_path[:-3]+'wav', win_len=4096, hop_len=512,
                              HPSS=False, only_bass=False)

            _chroma, _multi, _cnn, _rnn = BM.segment_bass(audio, midi, cnn_model, rnn_model, HPSS)
            results['chroma_bass_' + midi.name] = BM.get_performance(midi.gt_bass, _chroma)
            results['multi_bass_' + midi.name] = BM.get_performance(midi.gt_bass, _multi)
            results['cnn_bass_' + midi.name] = BM.get_performance(midi.gt_bass, _cnn)
            results['rnn_bass_' + midi.name] = BM.get_performance(midi.gt_bass, _rnn)

        pd_results = pd.DataFrame(results)
        pd_results.index = ['TP', 'FP', 'FN', 'recall', 'precision', 'f_score']
        pd_results = pd_results.T

        if HPSS:
            bass_file = ('/home/geduran/Environments/onsetDetection/Perfor' +
                         'mances/BassSegmentation_HPSS'+trainVal+mode+'.csv')
        else:
            bass_file = ('/home/geduran/Environments/onsetDetection/Perfor' +
                         'mances/BassSegmentation'+trainVal+mode+'.csv')

        all_performances = pd_results

        for ind in pd_results.index:
            if ind not in all_performances.index:
                all_performances = all_performances.append(pd_results.loc[ind])
        all_performances = all_performances.sort_index(axis=0)

        mid = int(len(all_performances.index) / num_files)

        methods = []


        for i in range(mid):
            methods.append(all_performances.iloc[i*num_files:(i+1)*num_files][['TP', 'FP', 'FN']].sum(axis=0))
            method = methods[i]
            methods[i]['recall'] = method['TP']/(method['TP']+method['FN'])
            methods[i]['precision'] = method['TP']/(method['TP']+method['FP'])
            methods[i]['f_score'] = 2 * method['precision']*method['recall'] / (method['precision'] + method['recall'])

        methods[0] = methods[0].rename('chroma_bass_TOTAL')
        methods[1] = methods[1].rename('cnn_bass_TOTAL')
        methods[2] = methods[2].rename('multi_bass_TOTAL')
        methods[3] = methods[3].rename('rnn_bass_TOTAL')

        all_performances = []
        for method in methods:
            all_performances.append(method)

        all_performances = pd.DataFrame(all_performances).round(2)
        print(trainVal)
        print(all_performances)
        all_performances.to_csv(bass_file, encoding='utf-8')
