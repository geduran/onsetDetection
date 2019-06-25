from audioData import *
from midiData import *
from dataManager import *
import glob
import warnings
import pandas as pd
import sys
import numpy         as     np
from   keras         import backend as K
import tensorflow    as     tf
from   keras.backend import tensorflow_backend
from   cnn_utils     import sequentialCNN, evaluateCNN, defineCallBacks, loadPatches, loadTestPatches, loadAudioPatches
from   cnn_utils     import evaluateLayer, plotCurves, computeConfussionMatrix, deleteWeights
from   rnn_utils     import sequentialRNN, evaluateRNN, defineCallBacks, loadAudioPatches
from   rnn_utils     import evaluateLayer, plotCurves, computeConfussionMatrix, deleteWeights, Metrics

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.filterwarnings(action="ignore", category=DeprecationWarning)

results = {}


HPSS = False

K.set_image_dim_ordering('th')
print(K.image_data_format())
config  = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
tensorflow_backend.set_session(session)

input_shape  = (2, 32, 128)
batch_size    = 128
droprate      = 0.25
num_classes  = 2
p             = [[15, 3], [5, 5], [3, 3]]   # Conv2D mask size
d             = [5, 4, 6]   # Conv2D channels
f             = [20]
n_hidden = 40
n_layers = 2


if len(sys.argv) >= 2:
    listItems = [sys.argv[1]]
else:
    listItems = ['1', '2', '3', 'all']


for mode in listItems:
    print('Analizando mode {}'.format(mode))
    #midi_files = glob.glob('/Users/gabrielduran007/Desktop/University/MAGISTER/codigos/RNN/MIDI/1/*.mid')
    midi_files = glob.glob('/home/geduran/Environments/MIDI/'+mode+'/*.mid')



    num_files = len(midi_files)

    #cnn_model_path = '/Users/gabrielduran007/Desktop/University/MAGISTER/codigos/CNN_audio/best_model_1.h5'
    cnn_model_path = '/home/geduran/Environments/CNN_audio/best_model_'+mode+'_cnnChord.h5'

    cnn_model = sequentialCNN(input_shape,droprate,num_classes,p,d,f)
    cnn_model.load_weights(cnn_model_path)

    input_shape  = (15, 256)
    #rnn_model_path = '/Users/gabrielduran007/Desktop/University/MAGISTER/codigos/RNN_audio/best_model_'+mode+'_rnn.h5'
    rnn_model_path = '/home/geduran/Environments/RNN_audio/best_model_'+mode+'_rnnChord.h5'
    rnn_model = sequentialRNN(input_shape,droprate,num_classes, n_hidden,n_layers=n_layers)
    rnn_model.load_weights(rnn_model_path)
    #rnn_model = RNN(32, 100, 2)
    #rnn_model.load_state_dict(torch.load('/Users/gabrielduran007/Desktop/University/'+
    #                'MAGISTER/codigos/RNN/trainedModels/model_1_chordMel19iters',
    #                map_location='cpu'))
    #rnn_model.load_state_dict(torch.load('/home/geduran/Environments/rnnTrain/bestModel_'+mode+'_chordMel',
    #                          map_location='cpu'))
    #rnn_model.eval()


    BM = ChordManager()

    #for midi_path in midi_files:
    for i in range(num_files):
        if i > len(midi_files)-1:
            break
        midi_path = midi_files[i]
        midi = MidiData(midi_path)
        audio= AudioData(midi_path[:-3]+'wav', win_len=1024, hop_len=128, HPSS=False, only_bass=False)

        _chroma, _multi, _cnn, _rnn = BM.segment_chord(audio, midi, cnn_model, rnn_model, HPSS)
        results['chroma_chord_' + midi.name] = BM.get_performance(midi.gt_chord, _chroma)
        results['multi_chord_' + midi.name] = BM.get_performance(midi.gt_chord, _multi)
        results['cnn_chord_' + midi.name] = BM.get_performance(midi.gt_chord, _cnn)
        results['rnn_chord_' + midi.name] = BM.get_performance(midi.gt_chord, _rnn)

    pd_results = pd.DataFrame(results)
    pd_results.index = ['TP', 'FP', 'FN', 'recall', 'precision', 'f_score']
    pd_results = pd_results.T

    if HPSS:
        chord_file = '/home/geduran/Environments/Performances/ChordSegmentation_HPSS'+mode+'.csv'
    else:
        chord_file = '/home/geduran/Environments/Performances/ChordSegmentation'+mode+'.csv'


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

    methods[0] = methods[0].rename('chroma_chord_TOTAL')
    methods[1] = methods[1].rename('cnn_chord_TOTAL')
    methods[2] = methods[2].rename('multi_chord_TOTAL')
    methods[3] = methods[3].rename('rnn_chord_TOTAL')

    all_performances = []
    for method in methods:
        all_performances.append(method)

    all_performances = pd.DataFrame(all_performances)

    print(all_performances)
    all_performances.to_csv(chord_file, encoding='utf-8')
