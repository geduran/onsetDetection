#from rnnManager import *
from midiData import *
from audioData import *
import glob
import os
import sys

def get_labels_samples(dir):
    files = glob.glob(dir + '*.mid')

    all_labels = np.array([])
    all_mel1 = np.array([])
    all_mel2 = np.array([])
    all_mel3 = np.array([])

    for file in files:
        curr_midi = MidiData(file)
        curr_audio = AudioData(file[:-4]+'.wav', win_len=4096, hop_len=256,
                               HPSS=False, only_bass=False)

        curr_mel1 = curr_audio.features.bass_mel_spectrogram1_cnn
        curr_mel2 = curr_audio.features.bass_mel_spectrogram2_cnn
        curr_mel3 = curr_audio.features.bass_mel_spectrogram3_cnn

        # print('mel1.shape: {}'.format(curr_mel1.shape))
        # print('mel2.shape: {}'.format(curr_mel2.shape))
        # print('mel3.shape: {}'.format(curr_mel3.shape))

        curr_labels = np.zeros((curr_mel1.shape[0], 1), dtype=int)

        curr_hop = curr_audio.hop_len
        curr_sr = curr_audio.sr

        for time in curr_midi.gt_bass:
            index = int(time*curr_sr/curr_hop)
            if index > len(curr_labels):
                continue
            curr_labels[index] = 1

        if (all_labels.any() and all_mel1.any() and all_mel2.any() and
            all_mel3.any()):
            all_labels = np.concatenate((all_labels, curr_labels), axis=0)
            all_mel1 = np.concatenate((all_mel1, curr_mel1), axis=0)
            all_mel2 = np.concatenate((all_mel2, curr_mel2), axis=0)
            all_mel3 = np.concatenate((all_mel3, curr_mel3), axis=0)
        else:
            all_labels = curr_labels
            all_mel1 = curr_mel1
            all_mel2 = curr_mel2
            all_mel3 = curr_mel3

    return all_labels, all_mel1, all_mel2, all_mel3

if len(sys.argv) > 2:
    i = sys.argv[1]
    directory = '/home/geduran/Environments/onsetDetection/MIDI/Train/' + i + '/'

    labels, mel1, mel2, mel3 = get_labels_samples(directory)

    file = open(directory + '/cnnBassData_cqt_mel.pkl', 'wb')
    pickle.dump((labels, mel1, mel2, mel3), file)
    file.close()
    print('\n\nEn '+ i +' tenemos Labels.size {} y samples.size {}'.format(labels.shape, mel1.shape))

else:
    for i in ['1', 'all']:
        directory = '/home/geduran/Environments/onsetDetection/MIDI/Train/' + i + '/'

        labels, mel1, mel2, mel3 = get_labels_samples(directory)

        file = open(directory + '/cnnBassData_cqt_mel.pkl', 'wb')
        pickle.dump((labels, mel1, mel2, mel3), file)
        file.close()

    #print('\n\nTenemos Labels.size {} y samples.size {}'.format(labels.shape, samples.shape))
        print('\n\nEn '+ i +' tenemos Labels.size {} y samples.size {}'.format(labels.shape, mel1.shape))



"""
midi_files = glob.glob('MIDI/*.mid')

midis = []

for file in midi_files:
    new_midi = MidiData(file)
    new_audio = AudioData(new_midi.path+'.wav', win_len=4096, hop_len=1024, HPSS=False)
    print('Midi {} tiene {} notas de bajo'.format(new_midi.name, len(new_midi.gt_bass)))
    midis.append(new_midi)
    plot_segmentation(new_midi, new_midi.gt_bass,'gt_bass')
"""
