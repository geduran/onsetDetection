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

        curr_mel1 = curr_audio.features.bass_mel_spectrogram1
        curr_mel2 = curr_audio.features.bass_mel_spectrogram2
        curr_mel3 = curr_audio.features.bass_mel_spectrogram3


        # print('mel1: {}, mel2: {}, mel3: {}'.format(curr_mel1.shape,
        #                                             curr_mel2.shape,
        #                                             curr_mel3.shape))

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

    directory = '../MIDI/Train/' + i + '/'
    labels, mels1, mels2, mels3 = get_labels_samples(directory)

    file = open(directory + '/rnnBassData_cqt_mel.pkl', 'wb')
    pickle.dump((labels, mels1, mels2, mels3), file)
    file.close()
    print('\n\nEn '+ i +' tenemos Labels.size {} y samples.size {}'.format(labels.shape, mels1.shape))

else:
    for i in ['1', 'all']:
        directory = '../MIDI/Train/' + i + '/'
        labels, mels1, mels2, mels3 = get_labels_samples(directory)

        file = open(directory + '/rnnBassData_cqt_mel.pkl', 'wb')
        pickle.dump((labels, mels1, mels2, mels3), file)
        file.close()
        print('\n\nEn '+ i +' tenemos Labels.size {} y samples.size {}'.format(labels.shape, mels1.shape))
