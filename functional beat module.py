import pydub
import wave
import librosa
import os
import csv
import numpy as np
import pyaudio
from threading import Timer
import time
from phue import Bridge
from scipy.ndimage import gaussian_filter1d
import random
import shutil
import sys

bridge = Bridge('10.0.0.10')
lights = [7,8,10,11,12,14,15]
spatial_lights = [[14, 10, 8, 15], [8, 14, 7, 15], [7, 8, 12, 15], [12, 7, 11, 15], [11, 12, 10, 15], [10, 14, 11, 15], [15, 14, 8, 7, 12, 11, 10, 15]]
spatial_lights_index = [14, 8, 7, 12, 11, 10, 15]


def sample(i):                                                                  #This function is what executes light commands
    command = lights_track[i][2]                                                #dictionary of command peices
    bridge.set_light(lights_track[i][1], command)


print('Converting...')
sound = pydub.AudioSegment.from_mp3(os.path.join('C:\\', 'Users', 'akauf', 'Desktop', 'song.mp3'))
sound.export(os.path.join('E:\\', 'Python_Projects', 'Audio_engine', 'temp.wav'), format="wav")
print('Converted File to wav!')
print('Horaay!')
print('I am coolguy')

#Loads audio into bits file
print('Extracting Data...')
y, sr = librosa.load(os.path.join(os.path.join('C:\\', 'Users', 'akauf', 'Desktop', 'song.mp3')), sr=44100)
y_g, sr_g = librosa.load(os.path.join(os.path.join('C:\\', 'Users', 'akauf', 'Desktop', 'song.mp3')), sr=11025)
print('Extracted!')

y_h, y_p = librosa.effects.hpss(y)
y_h_g, y_p_g = librosa.effects.hpss(y_g)

print('Seeking out dem beets...')
hop_length = 512
hop_length_g = 128
tempo, beats = librosa.beat.beat_track(y=y_p, sr=sr, hop_length=hop_length)
print('I reckon you got something like {:0.1f} beats per minute there'.format(tempo))
beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=hop_length)
print('Ima put this in a file or something')
librosa.output.times_csv(os.path.join('E:\\', 'Python_Projects', 'Audio_Engine', 'Logs', 'beat_times.csv'), beat_times)


def normalize(sequence, sigma=0, shift=True):
    values = []
    for i in sequence:
        if shift == True:
            values.append(float(i))
        if shift == False:
            values.append(float(abs(i)))
    values = gaussian_filter1d(values, sigma)
    low = min(values)
    high = max(values)
    if low > 0 or low == 0:
        factor = 255 / (high - low)
    elif low < 0:
        factor = 255 / (high + abs(low))
    for i in range(len(values)):
        if low < 0:
            values[i] += abs(low)
        elif low > 0:
            values[i] -= low
        values[i] *= factor
    newval = []
    for i in range(len(values)):
        newval.append([i, values[i]])
    return newval

arcgraph = normalize(y_g, sigma=2500, shift=False)
percgraph = normalize(y_p_g, sigma=250, shift=False)
harmgraph = normalize(y_h_g, sigma=1000)

beatlocation = 0
old_beats = []
new_beats = []
def percparse(graph, beat_times):
    print('Lemme holler at these beets...')
    perclist = []
    for i in beat_times:
        sample_time = graph[int((i * sr_g) + (hop_length_g / 2))][1]
        old_beats.append([i, sample_time])
        if sample_time > 85:
            new_beats.append([i, sample_time])
            sat = 255
            bri = int(sample_time)
            hue = random.randrange(0, 75000)
            global beatlocation
            fixture = spatial_lights[beatlocation][random.randrange(0, 4)]
            beatlocation = spatial_lights_index.index(fixture)
            perclist.append([i - .05, fixture, {'bri': bri, 'sat': sat, 'transitiontime': 1, 'hue': hue}, 'beat'])
    return perclist

lights_track = percparse(percgraph, beat_times)

def write_data(filename, graph):
    with open(filename, 'w') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(len(graph)):
            writer.writerow(graph[i])

write_data(os.path.join('E:\\', 'Python_Projects', 'Audio_Engine', 'Logs', 'oldbeats.csv'), old_beats)
write_data(os.path.join('E:\\', 'Python_Projects', 'Audio_Engine', 'Logs', 'newbeats.csv'), new_beats)
write_data(os.path.join('E:\\', 'Python_Projects', 'Audio_Engine', 'Logs', 'arcgraph.csv'), arcgraph)
write_data(os.path.join('E:\\', 'Python_Projects', 'Audio_Engine', 'Logs', 'percgraph.csv'), percgraph)
write_data(os.path.join('E:\\', 'Python_Projects', 'Audio_Engine', 'Logs', 'lights_track.csv'), lights_track)

def execute(wav):
    chunk = 1024
    wf = wave.open(wav, 'rb')
    p = pyaudio.PyAudio()

    stream = p.open(
        format = p.get_format_from_width(wf.getsampwidth()),
        channels = wf.getnchannels(),
        rate = wf.getframerate(),
        output = True)
    data = wf.readframes(chunk)

    print('running lights')
    for i in range(0, len(lights_track)):
        Timer(lights_track[i][0], sample, [i]).start()

    time.sleep(0.1)

    while data != '':
        stream.write(data)
        data = wf.readframes(chunk)

    stream.close()
    p.terminate()
bridge.set_light(lights, 'on', True)
execute(os.path.join('E:\\', 'Python_Projects', 'Audio_engine', 'temp.wav'))
