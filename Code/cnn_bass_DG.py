#from rnnManager import *
from midiData import *
from audioData import *
import glob
import os
import sys


def get_labels_samples(dir):
    files = glob.glob(dir + '*.mid')

    all_labels = np.array([])
    all_cqt = np.array([])
    all_mel = np.array([])

    for file in files:
        curr_midi = MidiData(file)
        curr_audio = AudioData(file[:-4]+'.wav', win_len=1024, hop_len=128, HPSS=False, only_bass=False)

        curr_cqt = curr_audio.features.bass_CQT
        print('curr_cqt S.shape {}'.format(curr_cqt.shape))
        curr_mel = curr_audio.features.bass_mel_spectrogram

        new_samples = np.zeros((curr_mel.shape[0], curr_cqt.shape[1]))
        for i in np.arange(0, curr_mel.shape[0]-1, 2):
            new_samples[i//2, :] = (curr_cqt[i,:] + curr_cqt[i+1,:])/2
        curr_cqt = new_samples

        print('curr_cqt S.shape {}'.format(curr_cqt.shape))
        print('curr_mel S.shape {}'.format(curr_mel.shape))

        curr_labels = np.zeros((curr_cqt.shape[1], 1), dtype=int)

        curr_hop = curr_audio.hop_len
        curr_sr = curr_audio.sr

        for time in curr_midi.gt_bass:
            index = int(time*curr_sr/curr_hop)
            if index > len(curr_labels):
                break
            curr_labels[index] = 1

        if all_labels.any() and all_cqt.any():
            all_labels = np.concatenate((all_labels, curr_labels), axis=0)
            all_cqt = np.concatenate((all_cqt, curr_cqt.T), axis=0)
            all_mel = np.concatenate((all_mel, curr_mel.T), axis=0)
        else:
            all_labels = curr_labels
            all_cqt = curr_cqt.T
            all_mel = curr_mel.T

        #print('labels.shape: {}\nsamples.shape: {}\n'.format(all_labels.shape, all_samples.shape))

    #means = np.mean(all_samples, axis=0)
    #stds = np.std(all_samples, axis=0)
    #all_samples = np.subtract(all_samples, means)
    #all_samples = np.divide(all_samples, stds)

    return all_labels, all_cqt, all_mel

if len(sys.argv) > 2:
    i = sys.argv[1]
    directory = '/home/geduran/Environments/onsetDetection/MIDI/Train/' + i + '/'

    labels, cqts, mels = get_labels_samples(directory)

    file = open(directory + '/cnnBassData_cqt_mel.pkl', 'wb')
    pickle.dump((labels, cqts, mels), file)
    file.close()
    print('\n\nEn '+ i +' tenemos Labels.size {} y samples.size {}'.format(labels.shape, cqts.shape))

else:
    for i in ['1', '2', '3', 'all']:
        directory = '/home/geduran/Environments/onsetDetection/MIDI/Train/' + i + '/'

        labels, cqts, mels = get_labels_samples(directory)

        file = open(directory + '/cnnBassData_cqt_mel.pkl', 'wb')
        pickle.dump((labels, cqts, mels), file)
        file.close()

    #print('\n\nTenemos Labels.size {} y samples.size {}'.format(labels.shape, samples.shape))
        print('\n\nEn '+ i +' tenemos Labels.size {} y samples.size {}'.format(labels.shape, cqts.shape))



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
