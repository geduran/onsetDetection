import scipy.io.wavfile
import scipy.signal
import scipy.misc
import librosa
import librosa.display
import collections
import pickle
import os
import madmom
from madmom.audio.filters   import MelFilterbank
import numpy                as np
import matplotlib.pyplot    as plt
import sklearn.cluster      as skl
from scipy.signal           import find_peaks
from midiData               import *


class Features:
    """
    Class that stores all the audio features. It has no methods.
    """
    def __init__(self, name, win_len=None, hop_len=None):
        self.name = name
        self.mfcc = None
        self.mfcc_flux = None
        self.chroma = None
        self.bass_chroma = None
        self.chroma_flux = None
        self.spectral_centroid = None
        self.spectral_flatness = None
        self.spectral_rolloff = None
        self.spectral_irregularity = None
        self.spectral_flux = None
        self.temprogram = None
        self.HFC = None
        self.rms = None
        self.tonnetz = None
        self.tonn_flux = None
        self.S_h = None
        self.S_p = None
        self.bass_CQT = None
        self.chord_CQT = None
        self.bass_mel_spectrogram = None
        self.chord_mel_spectrogram = None
        self.bass_mel_spectrogram1 = None
        self.bass_mel_spectrogram2 = None
        self.bass_mel_spectrogram3 = None


class AudioData:
    """
    Class that manages an audio file. It performs all the audio feature
    extraction. Those features can be saved and loaded from the objectFiles
    directory.
    """
    def __init__(self, path, win_len=None, hop_len=None, HPSS=True,
                 only_bass=False):
        self.path = path
        self.name = self.path.split('/')[-1][:-4]
        if os.path.isfile('objectFiles/' + self.name + '.pkl'):
            self.load()
            return None
        self.features = None
        self.win_len = None
        self.hop_len = None
        self.sr, audio = scipy.io.wavfile.read(self.path)
        audio = audio.astype('float32')

        if len(audio) < len(audio.flatten()):
            audio = (audio[:, 0] + audio[:, 1]) / 2

        if self.sr > 44100 and False:
            audio = librosa.core.resample(audio, self.sr, 44100)
            self.sr = 44100

        self.audio = audio
        self.bass_audio = None
        self.audio_h = None
        self.audio_p = None

        self.get_features(win_len=win_len, hop_len=hop_len, HPSS=HPSS,
                          only_bass=only_bass)
        #self.save()


    def get_features(self,win_len=2048, hop_len=512, HPSS=True, only_bass=False):
        self.win_len = win_len
        self.hop_len = hop_len
        feature = Features(self.name,  win_len, hop_len)
        print('\n')

        if HPSS:
            print('Efectuando HPSS a {}'.format(self.name), end="\r")
            self.audio_h, self.audio_p = librosa.effects.hpss(self.audio)
            scipy.io.wavfile.write('aux_audios/'+self.name + '_harmonic.wav',
                                   self.sr, self.audio_h/np.max(self.audio_h))
            scipy.io.wavfile.write('aux_audios/'+self.name + '_percussive.wav',
                                   self.sr, self.audio_p/np.max(self.audio_p))
        else:
            self.audio_h = self.audio
            self.audio_p = self.audio

        self.bass_audio = self.audio_h

        if only_bass:
            b, a = scipy.signal.butter(2, (20/(self.sr/2), 261/(self.sr/2)),
                                       btype='bandpass', analog=False,
                                       output='ba')
            self.bass_audio = scipy.signal.lfilter(b, a, self.audio_h)


        print('Efectuando stft harmonic a {}      '.format(self.name),
              end="\r")
        S_h, _ = librosa.magphase(librosa.core.stft(y=self.audio_h,
                                n_fft=win_len, hop_length=hop_len))
        S = S_h
        print('Efectuando stft Percussive a {}      '.format(self.name),
              end="\r")
        S_p, _ = librosa.magphase(librosa.core.stft(y=self.audio_p,
                                n_fft=win_len, hop_length=hop_len))
        feature.S_h = S_h
        feature.S_p = S_p


        print('Efectuando bass_CQT a {}      '.format(self.name), end="\r")
        feature.bass_CQT = np.abs(librosa.core.cqt(y=self.audio_h,sr=self.sr,
                                  hop_length=hop_len, n_bins=48,
                                  bins_per_octave=12))

        feature.bass_CQT_cnn = np.abs(librosa.core.cqt(y=self.audio_h,
                                      sr=self.sr, hop_length=hop_len,
                                      n_bins=96, bins_per_octave=24))


        feature.chord_CQT = np.abs(librosa.core.cqt(y=self.audio_h,sr=self.sr,
                                   hop_length=hop_len, n_bins=128,
                                   bins_per_octave=24))

        print('Efectuando mel1 a {}      '.format(self.name), end="\r")
        audioFrame = madmom.audio.signal.FramedSignal(self.audio,
                                                      frame_size=self.win_len,
                                                      hop_size=self.hop_len,
                                                      sample_rate=self.sr)

        feature.bass_mel_spectrogram1 = madmom.audio.FilteredSpectrogram(audioFrame,
                                        filterbank=MelFilterbank, num_bands=40,
                                        fmin=40, fmax=1000)

        feature.bass_mel_spectrogram1_cnn = madmom.audio.FilteredSpectrogram(audioFrame,
                                        filterbank=MelFilterbank, num_bands=39,
                                        fmin=40, fmax=2000)

        print('Efectuando mel2 a {}      '.format(self.name), end="\r")

        audioFrame = madmom.audio.signal.FramedSignal(self.audio,
                                                      frame_size=self.win_len/2,
                                                      hop_size=self.hop_len,
                                                      sample_rate=self.sr
                                                      )
        feature.bass_mel_spectrogram2 = madmom.audio.FilteredSpectrogram(audioFrame,
                                        filterbank=MelFilterbank, num_bands=40,
                                        fmin=40, fmax=1000)

        feature.bass_mel_spectrogram2_cnn = madmom.audio.FilteredSpectrogram(audioFrame,
                                        filterbank=MelFilterbank, num_bands=39,
                                        fmin=40, fmax=2000)

        print('Efectuando mel3 a {}      '.format(self.name), end="\r")
        audioFrame = madmom.audio.signal.FramedSignal(self.audio,
                                                      frame_size=self.win_len/4,
                                                      hop_size=self.hop_len,
                                                      sample_rate=self.sr)

        feature.bass_mel_spectrogram3 = madmom.audio.FilteredSpectrogram(audioFrame,
                                        filterbank=MelFilterbank, num_bands=40,
                                        fmin=40, fmax=1000)

        feature.bass_mel_spectrogram3_cnn = madmom.audio.FilteredSpectrogram(audioFrame,
                                        filterbank=MelFilterbank, num_bands=50,
                                        fmin=40, fmax=2000)



        # feature.chord_mel_spectrogram = librosa.feature.melspectrogram(S=S_h ,sr=self.sr)


        # librosa.display.specshow(librosa.power_to_db(feature.bass_mel_spectrogram,
        #                     ref=np.max), y_axis='mel',fmax=8000, x_axis='time')
        # plt.colorbar(format='%+2.0f dB')
        # plt.title('Mel spectrogram')
        # plt.tight_layout()
        # plt.savefig('BassMel_' + self.name + '.jpg')
        # plt.clf()
        #
        # librosa.display.specshow(librosa.amplitude_to_db(feature.bass_CQT, ref=np.max),
        #                         sr=self.sr, x_axis='time', y_axis='cqt_note')
        # plt.colorbar(format='%+2.0f dB')
        # plt.title('Constant-Q power spectrum')
        # plt.tight_layout()
        # plt.savefig('BassCqt_' + self.name + '.jpg')
        # plt.clf()

        print('Calculando chroma a {}      '.format(self.name), end="\r")
        feature.chroma = librosa.feature.chroma_cqt(y=self.audio_h, sr=self.sr,
                                                    hop_length=hop_len)

        print('Calculando bass chroma a {}      '.format(self.name), end="\r")
        feature.bass_chroma = librosa.feature.chroma_cqt(y=self.bass_audio,
                                                         sr=self.sr,
                                                         hop_length=hop_len)

        feature.chroma_flux = self.get_data_flux(feature.chroma)
        print('Calculando mfcc a {}      '.format(self.name), end="\r")
        feature.mfcc = librosa.feature.mfcc(sr=self.sr, S=S)
        feature.mfcc_flux = self.get_data_flux(feature.mfcc, dist=4)
        print('Calculando spec_centroid a {}      '.format(self.name),
              end="\r")
        feature.spectral_centroid = librosa.feature.spectral_centroid(sr=self.sr,
                                                                      S=S)
        print('Calculando spec_rolloff a {}      '.format(self.name),
              end="\r")
        feature.spectral_rolloff = librosa.feature.spectral_rolloff(sr=self.sr,
                                                                    S=S)
        print('Calculando spec_flatness a {}      '.format(self.name),
              end="\r")
        feature.spectral_flatness = librosa.feature.spectral_flatness(S=S)
        print('Calculando spec_irregularity a {}      '.format(self.name),
              end="\r")
        feature.spectral_irregularity = self.get_spectral_irregularity(S=S)
        print('Calculando spec_flux a {}      '.format(self.name), end="\r")
        feature.spectral_flux = self.get_spectral_flux(S=S_p)
        print('Calculando HFC a {}      '.format(self.name), end="\r")
        feature.HFC = self.get_HFC(S=S_p)
        print('Calculando spec_rms a {}      '.format(self.name), end="\r")
        feature.rms = librosa.feature.rmse(S=S_p, frame_length=win_len,
                                           hop_length=hop_len)
        print('Calculando spec_tonnetz a {}      '.format(self.name), end="\r")
        feature.tonnetz = librosa.feature.tonnetz(y=None, sr=self.sr,
                                                  chroma=feature.chroma)
        #feature.tonn_flux = self.get_data_flux(feature.tonnetz)**2

        self.features = feature
        print('Features de {} calculadas!                 '.format(self.name) +
              '                                           ', end="\r")


    def get_HFC(self, win_len=None, hop_len=None, S=None, y=None):
        if not S.any() and y.any() and hop_len and win_len:
            S, _ = librosa.magphase(librosa.core.stft(y=y,
                                    n_fft=win_len, hop_length=hop_len))
        x, y = S.shape
        #print('shape de S: {}'.format(S.shape))
        value = np.square(S)
        value = value.T * np.linspace(0, x, x)
        HFC = np.sum(value.T, axis=0)
        return np.nan_to_num(HFC)

    def get_spectral_flux(self, win_len=None, hop_len=None, S=None, y=None):
        if not S.any() and y.any() and hop_len and win_len:
            S, _ = librosa.magphase(librosa.core.stft(y=y,
                                    n_fft=win_len, hop_length=hop_len))
        x, y = S.shape
        #print('shape de S: {}'.format(S.shape))
        dif = np.square(np.subtract(S[:, 0:y-2], S[:, 1:y-1]))
        spec_flux = np.sum(dif, axis=0)
        spec_flux = np.pad(spec_flux, (1,1), 'edge')
        return np.nan_to_num(spec_flux)

    def get_spectral_irregularity(self, win_len=None, hop_len=None,
                                  S=None, y=None):
        if not S.any() and y.any() and hop_len and win_len:
            S, _ = librosa.magphase(librosa.core.stft(y=y,
                                    n_fft=win_len, hop_length=hop_len))
        x, y = S.shape
        dif = np.square(np.subtract(S[0:x-2,:], S[1:x-1,:]))
        spec_irregularity= np.divide(np.sum(dif, axis=0), np.sum(S, axis=0))
        return np.nan_to_num(spec_irregularity)

    def get_data_flux(self, data, dist=1):
        x, y = data.shape
        data_flux = np.subtract(data[:, dist:y], data[:, :y-dist])
        data_flux = np.pad(np.sum(data_flux, axis=0), (dist,dist), 'edge')
        pos = np.argmax(np.abs(data_flux))
        data_flux[pos] = 0.01
        return np.nan_to_num(data_flux)

    def norm_data(self, data, norm_type=0):
        if len(data) < len(data.flatten()):
            x, y = data.shape
        else:
            x = len(data)
            y = 1
        if norm_type:
            if x > y:
                data_norm = ((data - np.amin(data, axis=0)) /
                             (np.amax(data, axis=0) - np.amin(data, axis=0)))
            else:
                data_norm = ((data - np.amin(data, axis=1)) /
                             (np.amax(data, axis=1) - np.amin(data, axis=1)))
        else:
            if x > y:
                data_norm = (data - data.mean(axis=0)) / data.std(axis=0)
            else:
                data_norm = (data - data.mean(axis=1)) / data.std(axis=1)
        return data_norm


    def save(self):
        name = 'objectFiles/' + self.name + '.pkl'
        file = open(name, 'wb')
        pickle.dump(self.__dict__, file, 2)
        file.close()

    def load(self):
        name = 'objectFiles/' + self.name + '.pkl'
        file2 = open(name, 'rb')
        temp_dict = pickle.load(file2)
        file2.close()
        self.__dict__.update(temp_dict)
