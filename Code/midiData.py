import numpy as np
import music21
import random
from matplotlib import pyplot as plt
import scipy.io.wavfile
import math
from dataManager import *
from audioData import *

class Instrument:

    def __init__(self, name=None, notes=None):
        self.name = name
        self.notes = notes

class MidiData:

    def __init__(self, filename, generateWav=False):
        self.name = filename.split('/')[-1].split('.')[0].replace("""'""", '').replace('.', '')
        self.path = filename.split(filename.split('/')[-1])[0]
        print('\n\nIncluyendo midi '+ self.name)
        if not os.path.isfile(self.path + self.name + '.wav') and generateWav:
            self.generate_wav()
        self.instruments = []
        self.percussive_instruments = ['Closed Hi Hat', 'Tight Snare', 'Hi Ride Cymbal', 'Drums',
                                       'STANDARD Drum*gemischt', 'Side Stick', 'Loose Kick', 'Side Stick',
                                       'Shaker', 'Long Guiro', 'Short Guiro', 'Lo Conga', 'Open Hi Conga',
                                       'drums', 'StdDrums', 'SnareDrum', 'BassDrum',
                                       'Hihats', 'Toms', 'RideCymbal', 'CrashCymbal',
                                       'Rimshots', 'Hihatfills', 'Hi-hat', 'Tambourine',
                                       'Count-in', 'Tom', 'CrashCymbal', 'Snarefills',
                                       'cymbals', 'snare', 'bassdrum', 'ridecymbals', 'ShortGuiro',
                                       'LongGuiro', 'LoConga', 'LooseKick', 'OpenHiConga',
                                       'SideStick', 'Percussion', 'Morepercussion',
                                       'StdDrums', 'HandClaps', 'toms', 'TomToms',
                                       'openhihat1', 'KickDrum', 'Rhythm', 'OpenHiHat',
                                       'ClosedHiHat', 'TightSnare', 'Std.DrumSet', 'DRUMS',
                                       'Drums2', 'RhythmTrack', 'PercussionStdDrums',
                                       'Sandblock', '']

        self.colors = np.array([[255,0,0],[0,155,0],[0,0,255],[0,255,255],[255,0,255],[0,255,255],[100,255,100],
                               [100,100,255],[255,100,100],[100,100,100],
                               [200,200,200],[200,100,200],[50,50,50],[0,20, 120],[120,20,0]], dtype=np.uint8)

        self.note_names = ('A0', 'A0#', 'B0', 'C1', 'C1#', 'D1', 'D1#', 'E1', 'F1', 'F1#', 'G1', 'G1#',
                           'A1', 'A1#', 'B1', 'C2', 'C2#', 'D2', 'D2#', 'E2', 'F2', 'F2#', 'G2', 'G2#',
                           'A2', 'A2#', 'B2', 'C3', 'C3#', 'D3', 'D3#', 'E3', 'F3', 'F3#', 'G3', 'G3#',
                           'A3', 'A3#', 'B3', 'C4', 'C4#', 'D4', 'D4#', 'E4', 'F4', 'F4#', 'G4', 'G4#',
                           'A4', 'A4#', 'B4', 'C5', 'C5#', 'D5', 'D5#', 'E5', 'F5', 'F5#', 'G5', 'G5#',
                           'A5', 'A5#', 'B5', 'C6', 'C6#', 'D6', 'D6#', 'E6', 'F6', 'F6#', 'G6', 'G6#',
                           'A6', 'A6#', 'B6', 'C7', 'C7#', 'D7', 'D7#', 'E7', 'F7', 'F7#', 'G7', 'G7#',
                           'A7', 'A7#', 'B7', 'C7')
        self.tempo = None
        self.meter = None
        self.resolution = 120
        self.midi2song(filename)
        self.max_time = self.get_max_time()
        self.gt_bass = self.get_gt_bass()
        self.gt_beat = self.get_gt_beat()
        self.gt_chord = self.get_gt_chord()
        self.gt_note = self.get_gt_note()

    def generate_wav(self):
        outname = self.name + '.wav'
        os.system('fluidsynth -ni MIDIsounds/*sf2 '+ self.path +
                  self.name + '.mid -F '+ self.path + outname + ' -r 44100')

    def add_instrument(self, instrument):
        for inst in self.instruments:
            if instrument.name == inst.name:
                return None
    #    print('Agregamos ' + instrument.name)
        self.instruments.append(instrument)

    def get_max_time(self):
        max_time = 0
        for inst in self.instruments:
            for num in inst.notes.keys():
                if inst.notes[num]['end']:
                    max_time = max(max_time, max(inst.notes[num]['end']))
                    max_time = max(max_time, max(inst.notes[num]['start']))
        return max_time

    def midi2song(self, filename):
        mf = music21.midi.MidiFile()
        mf.open(filename)
        mf.read()
        mf.close()
        name = ''
        tempo = None
        for i in range(len(mf.tracks)):
            active = False
            notes = dict()
            for midi_note in range(21, 109):
                notes[str(midi_note)] = dict()
                notes[str(midi_note)]['start'] = list()
                notes[str(midi_note)]['end'] = list()

            event = mf.tracks[i].events
            t = 0
            for e in event:
                if 'SEQUENCE_TRACK_NAME' in str(e):
                    name = str(e).split("'")[1]
                elif 'TEXT_EVENT' in str(e):
                    name = str(e).split("'")[1]
                elif 'SET_TEMPO' in str(e):
                    data = str(e).replace('=','').split('\\x')
                    try:
                        if(len(data) > 1):
                            if len(data) > 3:
                                tempo = 256**2*int(data[1][:2], 16) + 256*int(data[2][:2], 16) + int(data[3][:2], 16)
                            elif len(data) > 2:
                                tempo = 256**2*int(data[1][:2], 16) + 256*int(data[2][:2], 16)
                            else:
                                tempo = 256**2*int(data[1][:2], 16)
                            tempo = int((10**6)/tempo * 60)
                    except:
                        tempo = 0
                    #print('tempo: ' + str(tempo))

                elif 'TIME_SIGNATURE' in str(e):
                    try:
                        data = str(e).split('\\x')
                        meter = str(int(data[1][:2])) + '/' + str(2**int(data[2][:2]))
                    except:
                        meter = '4/4'
                #    print('meter: ' + str(meter))

                if e.isDeltaTime and (e.time):
                #if ('DeltaTime' in str(e)) and (e.time is not None):
                    t += e.time
                elif(e.isNoteOn and (e.pitch) and (e.velocity)):
                    if e.pitch < 21 or e.pitch > 109:
                        continue
                #elif ('NOTE_ON' in str(e)) and (e.pitch):
                    notes[str(e.pitch)]['start'].append(t)
                    active = True
                elif(e.isNoteOn and (e.pitch) and not (e.velocity)):
                    if e.pitch < 21 or e.pitch > 109:
                        continue
                #elif ('NOTE_OFF' in str(e)) and (e.pitch):
                    notes[str(e.pitch)]['end'].append(t)


            name = name.replace(' ','')
            if (active): #and (name not in self.percussive_instruments) and ('Bater' not in name):
                self.add_instrument(Instrument(name=name, notes=notes))
        if not tempo:
            tempo = 1
        self.meter = meter
        self.tempo = tempo
        self.resolution = mf.ticksPerQuarterNote


    def get_gt_bass(self, low_limit=55): # G3!
        bass_gt = []
        for inst in self.instruments:
            if ((inst.name not in self.percussive_instruments) and
              ('Bater' not in inst.name) and ('bass' in inst.name or
               'Bass' in inst.name or 'bajo' in inst.name or 'Bajo' in inst.name)):
                for note in inst.notes.keys():
                    if int(note) <= low_limit:
                        for i in range(len(inst.notes[note]['start'])):
                            bass_gt.append(inst.notes[note]['start'][i]/(self.resolution*self.tempo)*60)
        bass_gt = np.sort(np.array(list(set(bass_gt))))

        i = 0
        min_dist = 1e-1
        while i < len(bass_gt)-1:
            if abs(bass_gt[i+1] - bass_gt[i]) < min_dist:
                bass_gt.pop(i)
                i -= 1
            i += 1

        return np.array(bass_gt)

    def get_gt_beat(self):
        return np.arange(0, self.max_time / self.resolution /(self.tempo/60), 60/self.tempo)

    def get_gt_chord(self, no_bass=True):

        chord_gt = []
        all_notes = {}
        #if no_bass:
    #        false_inst = ['Bass', 'bass', 'bajo', 'Bajo'] + self.percussive_instruments
#        else:
#            false_inst = self.percussive_instruments
        for inst in self.instruments:
            if inst.name not in self.percussive_instruments and 'Bater' not in inst.name:
                all_notes[inst.name] = []
                for note in inst.notes.keys():
                    for i in range(len(inst.notes[note]['start'])):
                        all_notes[inst.name].append(inst.notes[note]['start'][i]/(self.resolution*self.tempo)*60)


    #    for inst in self.instruments:
    #        if inst.name not in false_inst and 'Bater' not in inst.name:
    #            for note in inst.notes.keys():
    #                for i in range(len(inst.notes[note]['start'])):
    #                    all_notes.append(inst.notes[note]['start'][i]/(self.resolution*self.tempo)*60)
        for inst in all_notes.keys():
            all_notes[inst] = np.sort(np.array(all_notes[inst]))
        #all_notes = np.sort(np.array(all_notes))
            min_dist = 1e-2
            i = 1
            while i < len(all_notes[inst])-1:
                if (all_notes[inst][i+1] - all_notes[inst][i-1]) < min_dist:
                    chord_gt.append(all_notes[inst][i])
                    all_notes[inst] = np.delete(all_notes[inst], i+1)
                    all_notes[inst] = np.delete(all_notes[inst], i-1)
                    i -= 1
                i += 1

        i = 0
        min_dist = 1e-1
        while i < len(chord_gt)-1:
            if abs(chord_gt[i+1] - chord_gt[i]) < min_dist:
                chord_gt.pop(i)
                i -= 1
            i += 1

        return np.array(chord_gt)

    def get_gt_note(self):
        note_gt = []
        for inst in self.instruments:
            if inst.name not in self.percussive_instruments and 'Bater' not in inst.name:
                for note in inst.notes.keys():
                    for i in range(len(inst.notes[note]['start'])):
                        note_gt.append(inst.notes[note]['start'][i]/(self.resolution*self.tempo)*60)
        note_gt = np.array(list(set(note_gt)))
        note_gt = np.sort(note_gt)

        i = 0
        min_dist = 1e-1
        while i < len(note_gt)-1:
            if np.abs(note_gt[i+1] - note_gt[i]) < min_dist:
                note_gt = np.delete(note_gt, i)
                i -= 1
            i += 1

        return note_gt
