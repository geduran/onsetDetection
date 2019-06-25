from audioData import *
from midiData import *
from dataManager import *
import glob
import warnings
import os
import pandas as pd

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.filterwarnings(action="ignore", category=DeprecationWarning)

results = {}

midi_files = glob.glob('segmentation_files_/*.mid')

HPSS = False

BtM = BeatManager()

#for midi_path in midi_files:
for i in range(len(midi_files)):
    if i > len(midi_files)-1:
        break
    midi_path = midi_files[i]
    midi = MidiData(midi_path)
    audio = AudioData(midi_path[:-3]+'wav', win_len=1024, hop_len=256, HPSS=HPSS)
    BtM.rnn_segmentation(audio, '/Users/gabrielduran007/Desktop/University/RNN_example/trainedModels/modelo_beatSTFT103_iters')
    #_intensity, _tracker = BtM.segment_beat(audio, midi, HPSS)
    #results['beat_intensity_' + midi.name] = BtM.get_performance(midi.gt_beat, _intensity)
    #results['beat_tracker_' + midi.name] = BtM.get_performance(midi.gt_beat, _tracker)


pd_results = pd.DataFrame(results)
pd_results.index = ['TP', 'FP', 'FN', 'recall', 'precision', 'f_score']
pd_results = pd_results.T

if HPSS:
    beat_file = 'Performances/BeatSegmentation_HPSS.csv'
else:
    beat_file = 'Performances/BeatSegmentation.csv'

#if not os.path.isfile(beat_file):
#    pd_results.to_csv(beat_file, encoding='utf-8')

all_performances = pd.read_csv(beat_file, index_col = 0)

for ind in pd_results.index:
    if ind not in all_performances.index:
        all_performances = all_performances.append(pd_results.loc[ind])

all_performances = all_performances.sort_index(axis=0)

mid = int(len(all_performances.index) / 2)

method1 = all_performances.iloc[:mid][['TP', 'FP', 'FN']].sum(axis=0)
method2 = all_performances.iloc[mid:][['TP', 'FP', 'FN']].sum(axis=0)


method1['recall'] = method1['TP']/(method1['TP']+method1['FN'])
method1['precision'] = method1['TP']/(method1['TP']+method1['FP'])
method1['f_score'] = 2 * method1['precision']*method1['recall'] / (method1['precision'] + method1['recall'])

method2['recall'] = method2['TP']/(method2['TP']+method2['FN'])
method2['precision'] = method2['TP']/(method2['TP']+method2['FP'])
method2['f_score'] = 2 * method2['precision']*method2['recall'] / (method2['precision'] + method2['recall'])

method1 = method1.rename('beat_intensity_TOTAL')
method2 = method2.rename('beat_tracker_TOTAL')

all_performances = all_performances.append(method1)
all_performances = all_performances.append(method2)

print('all_performances: ')
print(all_performances)
#all_performances.to_csv(beat_file, encoding='utf-8')
