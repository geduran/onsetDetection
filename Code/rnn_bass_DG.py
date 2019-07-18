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
        curr_audio = AudioData(file[:-4]+'.wav', win_len=4096, hop_len=512,
                               HPSS=False, only_bass=False)

###########################################################
        # curr_cqt = curr_audio.features.bass_CQT
        # print('curr_cqt S.shape {}'.format(curr_cqt.shape))
        curr_mel1 = curr_audio.features.bass_mel_spectrogram1
        curr_mel2 = curr_audio.features.bass_mel_spectrogram2
        curr_mel3 = curr_audio.features.bass_mel_spectrogram3

    #    new_samples = np.zeros((curr_mel.shape[0], curr_cqt.shape[1]))
    #    for i in np.arange(0, curr_mel.shape[0]-1, 2):
    #        new_samples[i//2, :] = (curr_cqt[i,:] + curr_cqt[i+1,:])/2
    #    curr_cqt = new_samples
##########################################################


        # print('curr_cqt S.shape {}'.format(curr_cqt.shape))
        # print('curr_mel S.shape {}'.format(curr_mel1.shape))

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
            # print('allmel1.shape {}, allmel2.shape {}, allmel3.shape {}'.format(all_mel1.shape,all_mel2.shape,all_mel3.shape))
            # print('curr_mel1.shape {}, curr_mel2.shape {}, curr_mel3.shape {}'.format(curr_mel1.shape,curr_mel2.shape,curr_mel3.shape))
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
    labels, mels1, mels2, mels3 = get_labels_samples(directory)

    file = open(directory + '/rnnBassData_cqt_mel.pkl', 'wb')
    pickle.dump((labels, mels1, mels2, mels3), file)
    file.close()
    print('\n\nEn '+ i +' tenemos Labels.size {} y samples.size {}'.format(labels.shape, mels1.shape))

else:
    for i in ['1', 'all']:
        directory = '/home/geduran/Environments/onsetDetection/MIDI/Train/' + i + '/'
        labels, mels1, mels2, mels3 = get_labels_samples(directory)

        file = open(directory + '/rnnBassData_cqt_mel.pkl', 'wb')
        pickle.dump((labels, mels1, mels2, mels3), file)
        file.close()
        print('\n\nEn '+ i +' tenemos Labels.size {} y samples.size {}'.format(labels.shape, mels1.shape))



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
