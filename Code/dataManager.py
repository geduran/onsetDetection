from audioData import *
from midiData import *
import webcolors
#import torch
#import torch.nn as nn
#from rnnManager import *
from matplotlib import pyplot as plt


class DataManager:

    def __init__(self):
        pass

    def get_performance(self, ground_truth, detected, tolerance=5e-2):

        TP = 0
        FP = 0
        FN = 0

        total_gt = ground_truth.shape[0]
        total_det = detected.shape[0]

        for dt in detected:
            for gt in ground_truth:
                if np.abs(gt-dt) <= tolerance:
                    TP += 1
                    ground_truth = ground_truth[ground_truth!=gt]
                    break

        FN = total_gt - TP
        FP = total_det - TP
        print('TP: {}, FP: {}, FN: {}, total_gt {}, total_det: {}'.format(TP, FP, FN, total_gt, total_det))

        if TP:
            recall = TP/total_gt
            precision = TP/total_det
            f_score = 2 * (precision * recall) / (precision + recall)
        else:
            recall = 0
            precision = 0
            f_score = 0

        recall = round(recall, 2)
        precision = round(precision, 2)
        f_score = round(f_score, 2)

        print('recall {}, precision {}, f_score {}\n'.format(recall, precision, f_score))

        return [int(TP), int(FP), int(FN), recall,
                precision, f_score]


    def custom_segmentation(self, audioData, segments, seg_len=1, overlap=0):
        # Funcion para segmentar el audio en funcion de un arreglo de
        # segmentos previamente obtenidos. Se retorna una lista con cada
        # segmento. Los segmentos se pueden superponer en un porcentaje y
        # puede agarrar más de un segmento.
        seg_len -= 1
        segments = segments * audioData.sr
        segments = segments.astype(int)

        custom_seg = []
        start_ind = 0
        end_ind = segments[seg_len]

        segments = np.insert(segments, 0, 0)
        for i in range(seg_len-1):
            segments = np.insert(segments, 0, 0)

        start_cont = 0
        ref_start = segments[start_cont]

        end_cont = seg_len
        ref_end = segments[end_cont+1] - segments[end_cont]
        while end_ind <= len(audioData.audio):
            print('start_cont: {}, end_cont: {}, ref_start: {}, ref_end. {}'.format(start_cont, end_cont, ref_start, ref_end))
            print('start_ind {}, end_ind {}'.format(start_ind, end_ind))
            if np.abs(start_ind - segments[start_cont]) <= 4:
                start_ind = segments[start_cont]
                start_cont += 1
                ref_start = segments[start_cont] - segments[start_cont-1]

            if np.abs(end_ind - segments[end_cont+1]) <= 4:
                end_ind = segments[end_cont+1]
                end_cont += 1
                if end_cont+1 == len(segments):
                    ref_end = len(audioData.audio) - segments[end_cont]
                else:
                    ref_end = segments[end_cont+1] - segments[end_cont]

            custom_seg.append(self.window_segment(audioData.audio[start_ind:end_ind]))
            start_ind += int((1-overlap) * ref_start)
            end_ind += int((1-overlap) * ref_end)

        for i in range(len(custom_seg), len(segments)):
            custom_seg.append(self.window_segment(audioData.audio[segments[i]:]))
        return custom_seg

    def window_segment(self, segment):
        window = np.hamming(len(segment))
        windowed_function = window * segment
        return windowed_function

    def threshold_segmentation(self,audioData, feature="", thr=0.2, save_plot=False):
        path = '/Users/gabrielduran007/Desktop/University/MAGISTER/codigos/DataBase/Feature_plots/'

        curr_feature = np.atleast_2d(getattr(audioData.features, feature))
#                curr_feature = audioData.norm_data(np.abs(audioData.norm_data(curr_feature)), 'zero_one').T
        if 'chroma_flux' in curr_feature or 'mfcc_flux' in curr_feature:
            curr_feature = np.abs(self.norm_data(curr_feature)).T
        else:
            curr_feature = np.abs(audioData.norm_data(curr_feature, norm_type=1)).T

        #thr = 0.1 * np.max(curr_feature)
        peaks, _ = find_peaks(curr_feature.squeeze(), height=thr,
                              distance=20)
        if save_plot:
            plt.clf()
            if curr_feature.shape[1] > 1:
                plt.imshow(curr_feature)
            else:
                plt.plot(curr_feature)
            plt.plot(peaks, curr_feature[peaks], "x")
            plt.title(self.name+'_'+feature)
            plt.hlines(thr, 0, curr_feature.shape[0])
            plt.savefig(path+audioData.name+'_'+feature+'.eps', format='eps',
                        dpi=250)
        #print('Threshold Segmentation lista!                     ')
        # Transformamos las posiciones de indice de segmento a segundos
        thres_pos = (peaks * audioData.hop_len +
                          audioData.win_len/audioData.hop_len)/audioData.sr
        # Se devuelven las posiciones en segundos
        return thres_pos

    def cluster_segmentation(self, audioData, feature, save_plot=False):
        # Actualmente no lo estamos usando
        path = '/Users/gabrielduran007/Desktop/University/MAGISTER/codigos/DataBase/Feature_plots/'
        np_features = np.empty((0,audioData.features.chroma.shape[1]))
        curr_feature = np.atleast_2d(getattr(audioData.features, feature))
        np_features = np.concatenate((np_features, curr_feature),axis=0)

        np_features = np_features.T
        print('\n Empezando a clusterizar')
        clustering = skl.SpectralClustering(n_clusters=2, affinity='nearest_neighbors',
                                            n_neighbors=10).fit(np_features)
        #clustering = skl.KMeans(n_clusters=2, random_state=0).fit(np_features)
        labels = clustering.labels_
        pos = list()
        for i in range(len(labels)):
            if labels[i] == 0:
                pos.append(i)
        print('Clustering listo!               ', end = "\r")
        cluster_pos = (np.array(pos)  * audioData.hop_len +
                            audioData.win_len/audioData.hop_len)/audioData.sr
        return cluster_pos

    def similarity_clustering(self, audioData, name, pos):
        pos = ((audioData.sr * pos - audioData.win_len/audioData.hop_len) / audioData.hop_len).astype(int)
        chromas = np.empty((12,0))
        for i in range(len(pos)-1):
            curr_chroma = np.sum(audioData.features.chroma[:,pos[i]:pos[i+1]], axis=1).reshape((12,1))
            chromas = np.concatenate((chromas, curr_chroma), axis = 1)

        print('Iniciamos Clustering de Chromas')
        clustering = skl.SpectralClustering(n_clusters=7, affinity='nearest_neighbors',
                                            n_neighbors=10).fit(chromas.T)
        #clustering = skl.KMeans(n_clusters=2, random_state=0).fit(np_features)
        labels = clustering.labels_

        print('labels.shape: {}, pos.shape: {}'.format(labels.shape, pos.shape))
        colors = ['r', 'b', 'k', 'y', 'm', 'g', 'c']

        plt.clf()
        for i in range(len(labels)):
            pos1 = int((pos[i]* audioData.hop_len +
                   audioData.win_len/audioData.hop_len))

            pos2 = int((pos[i+1]* audioData.hop_len +
                   audioData.win_len/audioData.hop_len))

            t = np.linspace(pos1, pos2, pos2-pos1)
            plt.plot(t, audioData.audio[pos1:pos2], colors[labels[i]])

        plt.savefig('chroma_clustering.eps', format='eps', dpi=250)
        plt.clf()

        return labels

    def get_multipitch(self, audioData, win_len=4096, hop_len=1024, HPSS=False):

        if HPSS:
            audio = audioData.audio_h
        else:
            audio = audioData.audio

        audio = audio/max(audio)
        #scipy.io.wavfile.write('aux_audios/'+audioData.name + '_original.wav', audioData.sr, audio)

        b, a = scipy.signal.butter(2, (20/(audioData.sr/2), 261/(audioData.sr/2)),
                                btype='bandpass', analog=False, output='ba')
        audio = scipy.signal.lfilter(b, a, audio)

        audio = audio/max(audio)
        #scipy.io.wavfile.write('aux_audios/'+audioData.name + '_filtrado.wav', audioData.sr, audio)

        audio_slices = librosa.util.frame(audio, frame_length=win_len, hop_length=hop_len)
        hamming = np.tile(np.hamming(audio_slices.shape[0]), (audio_slices.shape[1],1))
        audio_slices = audio_slices * hamming.T

        # ACF con Wiener-Khinchin
        fft_audio = np.fft.fft(audio_slices, axis=0)
        xcorr_slices = np.fft.ifft(np.abs(fft_audio)**0.67, axis=0)

        # Consideramos lags positivos
        xcorr_slices = np.real(np.clip(xcorr_slices[0:int(xcorr_slices.shape[0]/2),:], 0 ,None))
        win_len = xcorr_slices.shape[0]
        xcorr_final = xcorr_slices


        for i in [2,3,4,5,7,8]:
            new_len = int(win_len/i) - 1
            xcorr_final -= np.clip(scipy.interpolate.pchip_interpolate(np.linspace(0, new_len, new_len),
                                                      xcorr_slices[0:new_len,:], np.linspace(0,new_len, win_len),
                                                      axis=0), 0, None)
            xcorr_final = np.clip(xcorr_final, 0, None)
            xcorr_final = np.absolute(np.clip(xcorr_final, 0, None))

        xcorr_final = np.absolute(np.clip(xcorr_final, 0, None))
        xcorr_final = scipy.ndimage.filters.gaussian_filter1d(xcorr_final, 3, axis=0)
        note_chroma = np.zeros((12, xcorr_final.shape[1]))
        cont = 0
        for x in xcorr_final.T:
            pos, _ = scipy.signal.find_peaks(x)
            #pos = self.get_true_maximum(x[pos-1], x[pos], x[pos+1])

            for p in pos:
                note = int(np.round(librosa.core.hz_to_midi(audioData.sr/p)))
                if note > 20 and note < 60:
                    note_chroma[note%12, cont] = x[p]
            cont += 1

        note_chroma = scipy.ndimage.filters.gaussian_filter1d(note_chroma, 3, axis=1)

        note_chroma_flux = np.clip(audioData.get_data_flux(note_chroma, dist=2),0 , None)**2

        note_chroma_flux = audioData.norm_data(note_chroma_flux, norm_type=1)

        pos, _ = scipy.signal.find_peaks(note_chroma_flux, distance=40,
                                         height=1.5e-2)

        pos = (pos * hop_len +
                          win_len/hop_len)/audioData.sr

        return pos

    def get_true_maximum(self, a ,b ,c):
        # get True maximum via Parabolic Interpolation
        return 0.5*(a-c)/(a - 2*b + c)

    def check_segmentation(self, audioData, name, pos):
        # Funcion para graficar y escuchar la segmentación sobre el audio.
        #plt.clf()
        #plt.plot(np.linspace(0, len(audioData.audio)/audioData.sr, len(audioData.audio)),
        #         audioData.audio)
        #plt.plot(pos, np.zeros(len(pos)), 'x')
        #plt.savefig('waveform_' + name + '.eps', format='eps', dpi=250)

        t = np.linspace(0, 0.05, (audioData.sr/20))
        click = 0.5*np.cos(2*np.pi*t*5e3)
        click_audio = np.copy(audioData.audio)/np.max(audioData.audio)

        for i in pos:
            i = int(i*audioData.sr)
            if i+int(audioData.sr/20) > len(click_audio):
                break
            click_audio[i:i+int(audioData.sr/20)] += 0.4*click
        scipy.io.wavfile.write('segmented_' + name + '.wav', audioData.sr, click_audio)

        print('En {} hay {} segmentos. '.format(name, len(pos)))


    def plot_piano_roll(self, midiData, save=True):
        duration = int(midiData.max_time/(midiData.resolution * midiData.tempo) * 60)
        piano_roll = np.ones([int(midiData.max_time/12), 88, 3], dtype=np.uint8)*255
        cont = 0
        for inst in midiData.instruments:
            if inst.name not in midiData.percussive_instruments and 'Bater' not in inst.name and 'bass' in inst.name or 'Bass' in inst.name:
                inst_color = midiData.colors[cont]
                cont += 1
                for note in inst.notes.keys():
                    for i in range(len(inst.notes[note]['start'])):
                        start_ = int(inst.notes[note]['start'][i]/12)
                        end_ = int(inst.notes[note]['end'][i]/12)
                        piano_roll[start_:end_, int(note)-21, :] = inst_color

        cont = 0
        piano_roll_ = np.repeat(piano_roll, 12, axis=1)
    #    piano_roll_[:, [(x-21)*12+6 for x in [43, 47, 50, 53, 57,
    #                                          64, 67, 71, 74, 77]], :] = 0
        piano_roll = np.transpose(piano_roll_, (1, 0, 2))
        plt.clf()
        #print('Piano.-roll: {}'.format(piano_roll.shape))
        plt.imshow(piano_roll[100:400,:int(midiData.resolution*midiData.tempo/(60)*10/12),:])
        plt.gca().invert_yaxis()

        x_pos = [x*midiData.resolution*midiData.tempo/(12*60) for x in range(0, 10)]
        x_label = [x for x in range(0, 10)]
        plt.xticks(x_pos, x_label)

        y_pos = [x*12+2 for x in range(0, 30)]
        plt.yticks(y_pos, midiData.note_names[10:40], fontsize='x-small')

        plt.savefig(midiData.name + '_segment.eps', format='eps', dpi = 200)
        plt.clf()

        y_pos = [x*12+6 for x in range(0, 89)]
        plt.tick_params(axis='both', labelsize=2)
        plt.yticks(y_pos, midiData.note_names)
        plt.title('PianoRoll de {}'.format(midiData.name), fontsize = 15)
        x_pos = [x*midiData.resolution*midiData.tempo/(12*60) for x in range(0, duration)]
        x_label = [x for x in range(0, duration)]
        plt.xticks(x_pos, x_label)
        if save:
            plt.savefig('../Segmentation_plots/{}_pianoRoll.eps'.format(midiData.name), format='eps', dpi=250)

    def plot_segmentation(self, midiData, segmentation_times,
                          segmentation_name, save=True):
        fc = 12
        #print('Entramos a plot_segmentation de ' + midiData.name)
        duration = int(midiData.max_time/(midiData.resolution * midiData.tempo) * 60)
        piano_roll = np.ones([int(midiData.max_time/fc), 88, 3], dtype=np.uint8)*255
        plt.clf()
        cont = 0
        for inst in midiData.instruments:
            if inst.name not in midiData.percussive_instruments and 'Bater' not in inst.name:
                inst_color = midiData.colors[cont]
                cont += 1
            #    print('En piano roll Graficamos ' + inst.name)
                for note in inst.notes.keys():
                    #print('starts tienen: {} y end tienen {}'.format(len(inst.notes[note]['start']) , len(inst.notes[note]['end'])))
                    if abs(len(inst.notes[note]['start']) - len(inst.notes[note]['end'])) > 0:
                        break
                    for i in range(len(inst.notes[note]['start'])):
                        start_ = int(inst.notes[note]['start'][i]/fc)
                        end_ = int(inst.notes[note]['end'][i]/fc)
                        piano_roll[start_:end_, int(note)-21, :] = inst_color
        #print('piano_roll.shape ' + str(piano_roll.shape))

        for i in segmentation_times:
            segm = int(i*midiData.resolution*midiData.tempo/(fc*60))
            plt.axvline(x=segm, color='k', linestyle='--', linewidth=0.1)
        cont = 0
        piano_roll_ = np.repeat(piano_roll, fc, axis=1)
        piano_roll = np.transpose(piano_roll_, (1, 0, 2))

        if not(piano_roll.shape[0] > 0 and piano_roll.shape[1] > 0 and piano_roll.shape[2] > 0):
            return None

        plt.imshow(piano_roll)
        y_pos = [x*12+6 for x in range(0, 89)]
        plt.tick_params(axis='both', labelsize=2)
        plt.gca().invert_yaxis()
        plt.hlines([(x-21)*12+6 for x in [43, 47, 50, 53, 57, 64,
                                          67, 71, 74, 77]], 0,
                    int(segmentation_times[-1] * midiData.resolution *
                    midiData.tempo/(12*60)), linewidth=0.4)
        plt.yticks(y_pos, midiData.note_names)
        x_pos = [x*midiData.resolution*midiData.tempo/(fc*60) for x in range(0, duration)]
        x_label = [x for x in range(0, duration)]
        plt.xticks(x_pos, x_label)
        plt.title('PianoRoll de {}, segmentado con {}'.format(
                   midiData.name, segmentation_name), fontsize = 7)
        path = midiData.path
        if '/' not in path[-1]:
            path += '/'
        if save:
            plt.savefig(path+'{}_pianoRoll_{}Segmented.eps'.format(midiData.name, segmentation_name), format='eps', dpi=250)
        #    plt.savefig('../Segmentation_plots/{}_pianoRoll_{}Segmented.eps'.format(midiData.name, segmentation_name), format='eps', dpi=250)

    def plot_cluster_segmentation(self, midiData, segmentation_times, clusters,
                                  segmentation_name, save=True):
        fc = 12
        duration = int(midiData.max_time/(midiData.resolution * midiData.tempo) * 60)
        piano_roll = np.ones([int(midiData.max_time/fc), 88, 3], dtype=np.uint8)*255

        for i in range(len(segmentation_times)-1):
            pos1 = int(segmentation_times[i]*midiData.resolution*midiData.tempo/(12*60))
            pos2 = int(segmentation_times[i+1]*midiData.resolution*midiData.tempo/(12*60))
            piano_roll[pos1:pos2, :, :] = (midiData.colors[clusters[i]]*2/3).astype(int)

        plt.clf()
        cont = 0
        for inst in midiData.instruments:
            if inst.name not in midiData.percussive_instruments:
                inst_color = midiData.colors[cont]
                cont += 1
                for note in inst.notes.keys():
                    for i in range(len(inst.notes[note]['start'])):
                        start_ = int(inst.notes[note]['start'][i]/fc)
                        end_ = int(inst.notes[note]['end'][i]/fc)
                        piano_roll[start_:end_, int(note)-21, :] = inst_color
        for i in segmentation_times:
            segm = int(i*midiData.resolution*midiData.tempo/(12*60))
            plt.axvline(x=segm, color='k', linestyle='--', linewidth=0.1)
        cont = 0
        piano_roll_ = np.repeat(piano_roll, 12, axis=1)

#        piano_roll_[:, [(x-21)*12+6 for x in [43, 47, 50, 53, 57, 64,
#                                              67, 71, 74, 77]], :] = 0
        piano_roll = np.transpose(piano_roll_, (1, 0, 2))

        piano_roll_warped = np.concatenate((piano_roll[:,0:799,:],piano_roll[:,800:800+799,:]), axis=0)

        piano_roll_warped = np.concatenate((piano_roll_warped, piano_roll[:,1600:1600+799,:]), axis=0)

        piano_roll_warped = np.concatenate((piano_roll_warped, piano_roll[:,0:799,:]), axis=0)

        plt.imshow(piano_roll_warped)
        plt.gca().invert_yaxis()
        plt.savefig('../Segmentation_plots/{}_pianoRoll_{}_clusterWARPED_Segmented.eps'.format(self.song.name, segmentation_name), format='eps', dpi=250)
        plt.clf()

        plt.imshow(piano_roll)

        y_pos = [x*12+6 for x in range(0, 89)]
        plt.tick_params(axis='both', labelsize=2)
        plt.gca().invert_yaxis()
        plt.hlines([(x-21)*12+6 for x in [43, 47, 50, 53, 57, 64,
                                          67, 71, 74, 77]], 0,
                    int(segmentation_times[-1] * midiData.resolution *
                    midiData.tempo/(12*60)), linewidth=0.4)

        plt.yticks(y_pos, midiData.note_names)

        x_pos = [x*midiData.resolution*midiData.tempo/(fc*60) for x in range(0, duration)]
        x_label = [x for x in range(0, duration)]
        plt.xticks(x_pos, x_label)

        plt.title('PianoRoll de {}, segmentado con {}'.format(
                   midiData.name, segmentation_name), fontsize = 7)
        if save:
            plt.savefig('../Segmentation_plots/{}_pianoRoll_{}_cluster_Segmented.eps'.format(self.song.name, segmentation_name), format='eps', dpi=250)

    def get_SDM(self, name, feature):
        path = '/Users/gabrielduran007/Desktop/University/MAGISTER/codigos/DataBase/SDM_plots/'
        plt.clf()
        curr_sdm = self_distance_matrix(feature, norm=2)
        plt.imshow(curr_sdm)
        plt.savefig(path+name[i]+'_SDM.eps', format='eps', dpi=250)
        plt.clf()


    def chord_cnn_segmentation(self, audio_data, model):

        file = open('/home/geduran/Environments/MIDI/Train/all/cnnChordData_cqt_mel.pkl', 'rb')
        #file = '/Users/gabrielduran007/Desktop/University/MAGISTER/codigos/RNN/1/BassData_mel.pkl'
        _,  _cqt, _mel = pickle.load(file)
        file.close()

        seq_len = 32
        cqt = audio_data.features.chord_CQT
        cqt = (cqt-np.min(_cqt)) / (np.max(_cqt)-np.min(_cqt))
        cqt = cqt.T
        n_samples = cqt.shape[0]
        n_features = cqt.shape[1]

        mel = audio_data.features.chord_mel_spectrogram
        mel = (mel-np.min(_mel)) / (np.max(_mel)-np.min(_mel))
        mel = mel.T

        evaluate_samples = np.zeros((n_samples-seq_len, 2, seq_len, n_features))
        for i in range(n_samples-seq_len):
            evaluate_samples[i,0,:,:] = cqt[i:i+seq_len,:]
            evaluate_samples[i,1,:,:] = mel[i:i+seq_len,:]

        batch_size = 1024
        predictions = model.predict(evaluate_samples, batch_size=batch_size, verbose=0)

        b, a = scipy.signal.butter(2, 0.8,btype='lowpass', analog=False, output='ba')
        detect_function = scipy.signal.lfilter(b, a, predictions[:,1])
        peaks, _ = scipy.signal.find_peaks(detect_function, height=0.5, distance=40)


        plt.clf()
        predictions_ = predictions[2000:5000,1]
        plt.plot(predictions_)
        peaks_, _ = scipy.signal.find_peaks(predictions_, height=0.5, distance=40)
        plt.plot(peaks_, predictions_[peaks_], 'x')
        plt.savefig(audio_data.name + '_chordCnn.eps', format='eps', dpi=100)
        plt.clf()

        peaks += int(seq_len/2)

        return (peaks * audio_data.hop_len +
                          audio_data.win_len/audio_data.hop_len)/audio_data.sr

    def chord_rnn_segmentation(self, audio_data, model):

        file = open('/home/geduran/Environments/MIDI/Train/all/rnnChordData_cqt_mel.pkl', 'rb')
        #file = '/Users/gabrielduran007/Desktop/University/MAGISTER/codigos/RNN/1/BassData_mel.pkl'
        _,  _cqt, _mel = pickle.load(file)
        file.close()

        seq_len = 30

        cqt = audio_data.features.chord_CQT
        cqt = (cqt-np.min(_cqt)) / (np.max(_cqt)-np.min(_cqt))
        cqt = cqt.T
        n_samples = cqt.shape[0]

        mel = audio_data.features.chord_mel_spectrogram
        mel = (mel-np.min(_mel)) / (np.max(_mel)-np.min(_mel))
        mel = mel.T

        curr_samples = np.concatenate((mel, cqt), axis=1)

        n_features = curr_samples.shape[1]

        evaluate_samples = np.zeros((n_samples-seq_len, seq_len, n_features))
        for i in range(n_samples-seq_len):
            evaluate_samples[i,:,:] = curr_samples[i:i+seq_len,:]

        #print('RNN- samples.shape {}'.format(evaluate_samples.shape))
        predictions = model.predict(evaluate_samples, batch_size=1024, verbose=0)
        #print('RNN- samples.shape {}'.format(predictions.shape))

        #predictions = predictions[2000:5000,:]
        #b, a = scipy.signal.butter(2, 0.9,btype='lowpass', analog=False, output='ba')
        #detect_function = scipy.signal.lfilter(b, a, predictions[:,1])
        peaks, _ = scipy.signal.find_peaks(predictions[:,1], height=0.5, distance=40)

        plt.clf()
        predictions_ = predictions[2000:5000,1]
        plt.plot(predictions_)
        peaks_, _ = scipy.signal.find_peaks(predictions_, height=0.85, distance=40)
        plt.plot(peaks_, predictions_[peaks_], 'x')
        plt.savefig(audio_data.name + '_chordRnn.eps', format='eps', dpi=100)
        plt.clf()

        return (peaks * audio_data.hop_len +
                          audio_data.win_len/audio_data.hop_len)/audio_data.sr


    def bass_cnn_segmentation(self, audio_data, model):

        file = open('/home/geduran/Environments/onsetDetection/MIDI/Train/all/cnnBassData_cqt_mel.pkl', 'rb')
        #file = '/Users/gabrielduran007/Desktop/University/MAGISTER/codigos/RNN/1/BassData_mel.pkl'
        _,  _cqt, _mel = pickle.load(file)
        file.close()

        seq_len = 32

        mel = audio_data.features.bass_mel_spectrogram
        cqt = audio_data.features.bass_CQT

        new_samples = np.zeros((mel.shape[0], cqt.shape[1]))
        for i in np.arange(0, mel.shape[0]-1, 2):
            new_samples[i//2, :] = (cqt[i,:] + cqt[i+1,:])/2
        cqt = new_samples

        cqt = (cqt-np.min(_cqt)) / (np.max(_cqt)-np.min(_cqt))
        cqt = cqt.T
        n_samples = cqt.shape[0]

        mel = (mel-np.min(_mel)) / (np.max(_mel)-np.min(_mel))
        mel = mel.T

        evaluate_samples = np.zeros((n_samples-seq_len, 2, seq_len, mel.shape[1]))
        for i in range(n_samples-seq_len):
            evaluate_samples[i,0,:,:] = cqt[i:i+seq_len,:]
            evaluate_samples[i,1,:,:] = mel[i:i+seq_len,:]

        predictions = model.predict(evaluate_samples, batch_size=1024, verbose=0)

        b, a = scipy.signal.butter(2, 0.8,btype='lowpass', analog=False, output='ba')
        detect_function = scipy.signal.lfilter(b, a, predictions[:,1])
        peaks, _ = scipy.signal.find_peaks(detect_function, height=0.5, distance=40)


        #plt.clf()
        #predictions_ = predictions[2000:5000,1]
        #plt.plot(predictions_)
        #peaks_, _ = scipy.signal.find_peaks(predictions_, height=0.5, distance=40)
        #plt.plot(peaks_, predictions_[peaks_], 'x')
        #plt.savefig(audio_data.name + '_bassCnn.eps', format='eps', dpi=100)
        #plt.clf()

        peaks += int(seq_len/2)

        return (peaks * audio_data.hop_len +
                          audio_data.win_len/audio_data.hop_len)/audio_data.sr

    def bass_rnn_segmentation(self, audio_data, model):

        file = open('/home/geduran/Environments/onsetDetection/MIDI/Train/all/rnnBassData_cqt_mel.pkl', 'rb')
        #file = '/Users/gabrielduran007/Desktop/University/MAGISTER/codigos/RNN/1/BassData_mel.pkl'
        _,  _cqt, _mel = pickle.load(file)
        file.close()

        seq_len = 30

        mel = audio_data.features.bass_mel_spectrogram
        cqt = audio_data.features.bass_CQT

        mx_mel = np.max(_mel)
        mn_mel = np.min(_mel)
        mel = (mel-mn_mel) / (mx_mel-mn_mel)

    #    new_samples = np.zeros((mel.shape[0], cqt.shape[1]))
    #    for i in np.arange(0, mel.shape[0]-1, 2):
    #        new_samples[i//2, :] = (cqt[i,:] + cqt[i+1,:])/2
    #    cqt = new_samples

        mx_cqt = np.max(_cqt)
        mn_cqt = np.min(_cqt)
        cqt = (cqt-mn_cqt) / (mx_cqt-mn_cqt)

        mel = mel.T
        cqt = cqt.T

        curr_samples = np.concatenate((mel, cqt), axis=1)

        n_samples = cqt.shape[0]
        n_features = curr_samples.shape[1]

        evaluate_samples = np.zeros((n_samples-seq_len, seq_len, n_features))
        for i in range(n_samples-seq_len):
            evaluate_samples[i,:,:] = curr_samples[i:i+seq_len,:]

        predictions = model.predict(evaluate_samples, batch_size=1024, verbose=0)

        #predictions = predictions[2000:5000,:]
        #b, a = scipy.signal.butter(2, 0.9,btype='lowpass', analog=False, output='ba')
        #detect_function = scipy.signal.lfilter(b, a, predictions[:,1])
        peaks, _ = scipy.signal.find_peaks(predictions[:,1], height=0.5, distance=40)

        #plt.clf()
        #predictions_ = predictions[2000:5000,1]
        #plt.plot(predictions_)
        #peaks_, _ = scipy.signal.find_peaks(predictions_, height=0.85, distance=40)
        #plt.plot(peaks_, predictions_[peaks_], 'x')
        #plt.savefig(audio_data.name + '_bassRnn.eps', format='eps', dpi=100)
        #plt.clf()

        #peaks += int(seq_len/2)

        return (peaks * audio_data.hop_len +
                          audio_data.win_len/audio_data.hop_len)/audio_data.sr


class BassManager(DataManager):

    def __init__(self):
        super().__init__()


    def segment_bass(self , audio, midi, cnn_model, rnn_model, HPSS=False):

        print('analizando ' + audio.name)
        chroma_segmentation = self.chroma_segmentation(audio,
                                         hop_len=512, HPSS=HPSS)

        multipitch_segmentation = self.get_multipitch(audio,
                                             win_len=2048,hop_len=256,
                                             HPSS=HPSS)
        cnn_segmentation = self.bass_cnn_segmentation(audio, cnn_model)

        rnn_segmentation = self.bass_rnn_segmentation(audio, rnn_model)

        self.check_segmentation(audio, audio.name +'bassChroma', chroma_segmentation)
        self.check_segmentation(audio, audio.name +'bassMulti', multipitch_segmentation)
        self.check_segmentation(audio, audio.name +'bassCNN', cnn_segmentation)
        self.check_segmentation(audio, audio.name +'bassRNN', rnn_segmentation)
        self.check_segmentation(audio, audio.name +'bassGT', midi.gt_bass)

        #print('gt: {}'.format(midi.gt_bass[:25]))
        #print('RNN: {}'.format(rnn_segmentation[:25]))

        #gt = midi.gt_bass

        try:
            self.plot_segmentation(midi, midi.gt_bass,'gt_bass')
        except:
            print('NO SE PUDO PLOTEAR...')

        if True:
            self.plot_segmentation(midi, chroma_segmentation,
                                   'Chroma_bass')
            self.plot_segmentation(midi, multipitch_segmentation,
                                  'Multi_bass')
            self.plot_segmentation(midi, cnn_segmentation,
                                  'CNN_bass')
            self.plot_segmentation(midi, rnn_segmentation,
                                  'RNN_bass')

        return chroma_segmentation, multipitch_segmentation, cnn_segmentation, rnn_segmentation


    def chroma_segmentation(self, audio_data, hop_len=1024, HPSS=False):

        factor = 1 # Numero de bins por nota
        if HPSS:
            audio = audio_data.audio_h
        else:
            audio = audio_data.audio

        audio = audio/max(audio)

        b, a = scipy.signal.butter(2, (20/(audio_data.sr/2), 261/(audio_data.sr/2)),
                                btype='bandpass', analog=False, output='ba')
        audio = scipy.signal.lfilter(b, a, audio)


        # Se busca hasta el C4
        cqt_t = librosa.cqt(audio, sr=audio_data.sr, hop_length=hop_len,
                            bins_per_octave=int(12*factor), n_bins=int(36*factor)+1,
                            window='hamm')

        C_new = np.abs(cqt_t)
        CQT = librosa.amplitude_to_db(C_new, ref=np.max)

        C_new = scipy.ndimage.morphology.grey_closing(C_new, size=(1,10))
        C_new = scipy.ndimage.filters.gaussian_filter1d(C_new, 4, axis=1)

        suma = np.clip(audio_data.get_data_flux(C_new, dist=2), 0, None)**2
        suma = audio_data.norm_data(suma, norm_type=1)

        thr = 0.005 * np.max(suma)
        pos, _ = scipy.signal.find_peaks(suma, distance=1, height=thr)
        #plt.clf()
        #tt = np.linspace(0, 10, len(suma[:int(10/hop_len*audio_data.sr)]))
        #plt.plot(tt, suma[:int(10/hop_len*audio_data.sr)])
        #plt.plot(suma)
        #plt.plot(pos, suma[pos], 'x')
    #    plt.plot(pos[:int(10*hop_len/audio_data.sr)], suma[pos[:int(10*hop_len/audio_data.sr)]], 'x')
        #plt.savefig(audio_data.name+'_chroma_seg.eps', format='eps', dpi=200)
        #plt.clf()
        pos = ((pos + 1) * hop_len) / audio_data.sr
        return pos



class ChordManager(DataManager):

    def __init__(self):
        super().__init__()

    def segment_chord(self, audio, midi, cnn_model, rnn_model, HPSS=False):

        print('analizando ' + audio.name)
        chroma_flux = self.threshold_segmentation(audio,
                                feature='chroma_flux', thr=0.1)

        harmonic_flux = self.chord_segmentation(audio,
                                             hop_len=audio.hop_len,
                                             HPSS=HPSS, thr=0.005)

        cnn_segmentation = self.chord_cnn_segmentation(audio, cnn_model)

        rnn_segmentation = self.chord_rnn_segmentation(audio, rnn_model)

        self.check_segmentation(audio, audio.name +'chordThresh', chroma_flux)
        self.check_segmentation(audio, audio.name +'chordTonnetz', harmonic_flux)
        self.check_segmentation(audio, audio.name +'chordCNN', cnn_segmentation)
        self.check_segmentation(audio, audio.name +'chordRNN', rnn_segmentation)
        self.check_segmentation(audio, audio.name +'chordRNN', rnn_segmentation)


        try:
            self.plot_segmentation(midi, midi.gt_chord, 'gt_chord')
        except:
            print('NO SE PUDO PLOTEAR...')

        if True:
            self.plot_segmentation(midi, chroma_flux,
                                   'Chroma_chord')
            self.plot_segmentation(midi, harmonic_flux,
                                  'Tonnetz_chord')
            self.plot_segmentation(midi, cnn_segmentation,
                                  'CNN_chord')
            self.plot_segmentation(midi, rnn_segmentation,
                                  'RNN_chord')

        return chroma_flux, harmonic_flux, cnn_segmentation, rnn_segmentation


    def chord_segmentation(self, audio_data, onsets=np.array([]), hop_len=512, thr=0.05,
                                 HPSS=False):
        # Se tiene la opcion de segmentarlo por segmentos regulares o de
        # entregarle un arreglo de segmentos
        path = '/Users/gabrielduran007/Desktop/University/MAGISTER/codigos/DataBase/Feature_plots/'
        if HPSS:
            audio = audio_data.audio_h
        else:
            audio = audio_data.audio

        if not onsets.any():
            #audio = audio_data.audio_h

###########
            #print('Caculamos CQT con menor hop_len')
            #cqt =librosa.feature.chroma_cqt(y=audio, sr=audio_data.sr,
            #                                hop_length=hop_len)
            cqt = audio_data.features.chroma
            #cqt = np.sum(cqt, axis=1, keepdims=True) / cqt.shape[1]
            ton = librosa.feature.tonnetz(y=None,
                                          sr=audio_data.sr, chroma=cqt)
############


            tonn_flux = audio_data.get_data_flux(ton, dist=1)**2
            #plt.clf()
            #plt.plot(tonn_flux[:300])
            #plt.savefig('Tonn_Flux.eps', format='eps', dpi=250)
            #plt.clf()

            curr_feature = tonn_flux
            #print('tonnetz.shape: {}'.format(tonn_flux.shape))
            #curr_feature = audio_data.norm_data(np.atleast_2d(tonn_flux))
            #curr_feature = audio_data.norm_data(np.abs(audio_data.norm_data(curr_feature)), 'zero_one').T

            #b, a = scipy.signal.butter(2, 0.7, btype='lowpass',
            #                           analog=False, output='ba')
            #curr_feature = scipy.signal.lfilter(b, a, curr_feature)
            #thr = 0.05 * np.max(curr_feature)
            peaks, _ = find_peaks(curr_feature.squeeze(), height=thr, distance=40)

            #plt.plot(curr_feature)
            #plt.plot(peaks, curr_feature[peaks], "x")
            #plt.title(audio_data.name+'_bassThres')
            #plt.hlines(thr, 0, curr_feature.shape[0])
            #plt.savefig(path+audio_data.name+'_ChordThres'+'.eps', format='eps', dpi=250)
            #print('Chord Threshold Segmentation lista!                   ')
            thres_pos = (peaks * hop_len +
                         audio_data.win_len/hop_len)/audio_data.sr
        else:
            custom_seg = audio_data.custom_segmentation(onsets, overlap=0.5)
            audio_data.feature.tonnetz = np.empty((6, 1))
            for seg in custom_seg:
                cqt =librosa.feature.chroma_cqt(y=seg, sr=audio_data.sr,
                                                hop_length=2**20)
                #cqt = np.sum(cqt, axis=1, keepdims=True) / cqt.shape[1]
                ton = librosa.feature.tonnetz(y=None,
                                              sr=audio_data.sr, chroma=cqt)
                audio_data.feature.tonnetz = np.concatenate((audio_data.feature.tonnetz,
                                                       ton), axis=1)

            audio_data.feature.tonnetz = audio_data.feature.tonnetz[:,1:]
            #print('tonnetz tieen shape: ' + str(audio_data.feature.tonnetz.shape))
            tonn_flux = audio_data.get_data_flux(audio_data.feature.tonnetz, dist=2)#np.clip(audio_data.get_data_flux(audio_data.feature.tonnetz, dist=1),
                                              #0, None)**2
            #plt.clf()
            #plt.plot(tonn_flux[:300])
            #plt.savefig('Tonn_Flux.eps', format='eps', dpi=250)
            #plt.clf()

            curr_feature = tonn_flux
            #curr_feature = audio_data.norm_data(np.atleast_2d(tonn_flux))

            peaks, _ = find_peaks(curr_feature.squeeze(), height=thr,
                                  distance=2)
            #print('Separacion es: {}'.format(int(audio_data.sr/(hop_len*4))))
            #plt.plot(curr_feature)
            #plt.plot(peaks, curr_feature[peaks], "x")
            #plt.title(audio_data.name+'_bassThres')
            #plt.hlines(thr, 0, curr_feature.shape[0])
            #plt.savefig(path+audio_data.name+'_ChordThres'+'.eps', format='eps', dpi=250)
            #print('Chord Threshold Segmentation lista!                   ')
            thres_pos = onsets[peaks]
        return thres_pos


class BeatManager(DataManager):

    def __init__(self):
        super().__init__()

    def segment_beat(self, audio_data, midi, HPSS=False):
        print('analizando ' + audio_data.name)
        if HPSS:
            audio = audio_data.audio_p
        else:
            audio = audio_data.audio

        beat_intensity = self.intensity_segmentation(audio_data)

        beat_tracker = self.get_beat_tracker(audio, sr=audio_data.sr, start=midi.tempo-10)
        gt = midi.gt_beat

        try:
            self.plot_segmentation(midi, gt, 'gt_beat')
        except:
            print('NO SE PUDO PLOTEAR...')


        if True:
            self.plot_segmentation(midi, beat_intensity,
                                   'beat_intensity')
            self.plot_segmentation(midi, beat_tracker,
                                  'beat_tracker')

        return beat_intensity, beat_tracker

    def get_beat_tracker(self, audio, sr=44100, start=60):
        tempo, beat_times = librosa.beat.beat_track(audio,
                            sr=sr, start_bpm=start, units='time')
        return beat_times

    def intensity_segmentation(self,audio_data,  thr=0.05):
        pos_beat = self.threshold_segmentation(audio_data, feature='HFC',
                                               thr=thr)
        return pos_beat
