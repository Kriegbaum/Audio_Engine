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
spatial_lights = [[14, 10, 15, 8], [15, 14, 7, 8], [7, 15, 12, 8], [12, 7, 11, 8], [11, 12, 10, 8], [10, 14, 11, 8], [8, 14, 8, 7, 12, 11, 10, 15]]
spatial_lights_index = [14, 8, 7, 12, 11, 10, 15]


def sample(i):                                                                  #This function is what executes light commands
    command = lights_track[i][2]                                                #dictionary of command peices
    bridge.set_light(lights_track[i][1], command)

def hue_process_i(value):
    if value > 255:
        value = 255
    if value < 1:
        value = 1
    if value > 128:
        value = value
        if value > 255:
            value = 255
    if value < 128:
        value = value
    value = 255 - value
    hue = abs(value * 200)
    return hue

print('Converting...')
sound = pydub.AudioSegment.from_mp3(os.path.join('C:\\', 'Users', 'akauf', 'Desktop', 'song.mp3'))
sound.export(os.path.join('E:\\', 'Python_Projects', 'Audio_engine', 'temp.wav'), format="wav")
print('Converted File to wav!')
print('Horaay!')
print('I am coolguy')

#Loads audio into bits file
print('Extracting Data...')
y, sr = librosa.load(os.path.join(os.path.join('C:\\', 'Users', 'akauf', 'Desktop', 'song.mp3')), sr=44100)
y_g, sr_g = librosa.load(os.path.join(os.path.join('C:\\', 'Users', 'akauf', 'Desktop', 'song.mp3')), sr=689)
print('Extracted!')

print('Peforming harmonic and percussive seperation...')
y_h, y_p = librosa.effects.hpss(y)
y_h_g, y_p_g = librosa.effects.hpss(y_g)
print('Success!')

print('Seeking out dem beets...')
hop_length = 512
tempo, beats = librosa.beat.beat_track(y=y_p, sr=sr, hop_length=hop_length)
print('I reckon you got something like {:0.1f} beats per minute there'.format(tempo))
beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=hop_length)
print('Ima put this in a file or something')
librosa.output.times_csv(os.path.join('E:\\', 'Python_Projects', 'Audio_Engine', 'Logs', 'beat_times.csv'), beat_times)
sr_g = float(sr_g)
hop_length_g = 0.011 * sr_g


def normalize(sequence, sigma=0, shift=True):
    print('Im doing some normalization')
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

def sumgraph(graph, sigma=15):                                      #Akin to position/time graph, takes a list of frames, creates a graph
#Effect refers to what item in N is used, sigma is the sigma value for gaussian smoothing
    print('Im summing up stuff')
    avg_tmp = 0
    for i in range(len(graph)):                                       #Grab the average power for given N
        avg_tmp += graph[i]
    avg = avg_tmp / len(graph)

    run = avg                                                                   #run is a value that gets modified each frame
    newgraph = []                                                                  #[x, y] positions for the sumgraph
    #CAN BE DONE WITH LAMBDA, PHASE THIS OUT
    graph_values = []                                                           #Sumgraph values without x value
    for i in range(len(graph)):
        if graph[i] < 0.1 and i != 0:
            graph[i] = abs(avg - 2)
        run += graph[i] - avg                                         #Modifies run up or down depending on how far off average it is
        newgraph.append([i, 0])
        #REFERENCE TO FEATURE PLANNED FOR DELETION
        graph_values.append(run)

    graph_min = abs(min(graph_values))                                          #Determines lowest point of graph
    graph_max = max(graph_values)                                               #Determines highest point of graph
    factor = 255 / (graph_max + graph_min)                                      #Determines mulitplying factor for next function

    for i in range(len(graph)):
        graph_values[i] += graph_min                                            #Raises graph so lowest point is now zero
        graph_values[i] *= factor                                               #Readjusts graph so that highest point is 255

    graph_values = gaussian_filter1d(graph_values, sigma)                       #Applies gaussian smoothing to graph, using supplied sigma

    for i in range(len(graph_values)):
        newgraph[i][1] = graph_values[i]
    return newgraph

def derive(graph, shift=True):                                                  #Makes a derivative of the supplied sumgraph
#Shift determines whether or not the graph is scaled to 0-255
    print('Im drinking and deriving')
    deriv_values = []                                                           #Y values
    deriv_graph = []                                                            #X and Y values
    for i in range(len(graph) - 1):                                             #Calculates slope at each frame, skips last frame
        slope = (graph[i + 1][1] - graph[i][1]) / (graph[i + 1][0] - graph[i][0])
        deriv_graph.append([i, slope])
        deriv_values.append(slope)
    deriv_max = max(deriv_values)                                               #Max and min are used for scaling the graph
    deriv_min = abs(min(deriv_values))
    if shift == True:
        factor = 255 / (deriv_max + deriv_min)
    else:
        factor = 255 / (deriv_max)
    for i in range(len(deriv_graph)):
        if shift == True:
            deriv_graph[i][1] += deriv_min
        deriv_graph[i][1] *= factor
    return deriv_graph


arcgraph = normalize(y_g, sigma=(1.85 * sr_g), shift=False)
percgraph = normalize(y_p_g, sigma=(0.0224 * sr_g), shift=False)
harmgraph = normalize(y_h_g, sigma=(.9571 * sr_g))
largearc = sumgraph(y_g, sigma=(5.442 * sr_g))

def fix(graph):
    print('Fixing these whack-ass harmonies...')
    newgraph = []
    hop = int(hop_length_g * 10)
    for i in range(hop, len(graph), hop):
        run = [x[1] for x in harmgraph[i - hop:]]
        run = gaussian_filter1d(run, 25)
        tmp = 0
        for s in range(i - hop, i + hop - 1):
            if tmp == len(run):
                break
            else:
                graph[s][1] = run[tmp]
            tmp += 1
    lrg = [x[1] for x in harmgraph]
    lrg = gaussian_filter1d(lrg, .6 * sr_g)
    for i in range(len(harmgraph)):
        harmgraph[i][1] = lrg[i]
    print('Fixed')

fix(harmgraph)

def arcparse(graph):
    print('Parsing Arc...')
    zero_list = []
    deriv = derive(graph, False)
    for i in range(len(deriv) - 1):
        if (deriv[i][1] < 0 and deriv[i + 1][1] >= 0) or (deriv[i][1] > 0 and deriv[i + 1][1] <= 0):
            zero_list.append(graph[i])

    for i in range(len(zero_list) - 1):
        gap = zero_list[i + 1][0] - zero_list[i][0]
        zero_list[i].append(gap)
    arclist = []
    for i in range(len(zero_list) - 1):
        if len(zero_list[i]) == 3:
            nex = zero_list[i + 1][0]
            hue = hue_process_i(graph[nex][1])
            sat = 255
            bri = int((largearc[nex][1] + graph[nex][1]) / 2)
            trans =  int((zero_list[i][2] / sr_g) * 10)
            if trans == 0:
                continue
            arclist.append([zero_list[i][0] / sr_g, lights, {'bri': bri, 'sat': sat, 'transitiontime': trans, 'hue': int(hue)}, 'arc'])
    print('Success!')
    return arclist

beatlocation = 0

def percparse(graph, beat_times):
    print('Lemme holler at these beets...')
    perclist = []
    inc = 0
    for i in beat_times:
        inc += 1
        sample_time = graph[int((i * sr_g) + (hop_length_g / 2))][1]
        if sample_time > 87:
            sat = 128 + random.randrange(0, 128)
            if inc % 2 == 0:
                bri = int(sample_time) + 90
                if bri > 254:
                    bri = 254
            else:
                bri = int(sample_time) - 90
                if bri < 0:
                    bri = 0
            hue = int(percgraph[int(i * sr_g)][1])
            global beatlocation
            fixture = spatial_lights[beatlocation][random.randrange(0, 4)]
            beatlocation = spatial_lights_index.index(fixture)
            perclist.append([i - .05, fixture, {'bri': bri, 'sat': sat, 'transitiontime': 1, 'hue': hue}, 'beat'])
    return perclist

def harmparse(graph):
    print('Parsing Harmonies...')
    deriv = derive(graph, False)
    low = []
    high = []
    combined = []
    for i in range(len(deriv) - 1):
        if (deriv[i][1] < 0 and deriv[i + 1][1] >= 0 and largearc[i][1] > 20):
            low.append(deriv[i] + ['low'])
        elif (deriv[i][1] > 0 and deriv[i + 1][1] <= 0 and largearc[i][1] > 20):
            high.append(deriv[i] + ['high'])
    for i in range(min([len(low), len(high)])):
        combined.append(low.pop(0))
        try:
            if high[0][0] > combined[-1][0]:
                combined.append(high.pop(0))
            elif high[1][0] > combined[-1][0]:
                combined.append(high.pop(1))
                del high[0]
        except:
            continue
    for i in range(len(combined) - 1):
        combined[i][1] = combined[i + 1][1]
        gap = combined[i + 1][0] - combined[i][0]
        gap = (gap * 10) / sr_g
        if gap == 0:
            gap = 'DEL'
        combined[i].append(gap)
        combined[i].append(graph[combined[i][0]][1])
    for i in range(len(combined) - 1):
        if combined[i][2] == 'low':
            bri = 140 + graph[i][1] * 1.25
            if bri > 254:
                bri = 254
            if bri < 0:
                bri = 0
            combined[i][1] = bri
        if len(combined[i]) > 4:
            sat = 255
            combined[i][4] = sat
            hue = int(hue_process_i(arcgraph[i][1] * .5 + .5 * graph[i][1]))
            combined[i].append(hue)
    harmlist = []
    for i in range(0, len(combined) - 1, 2):
        global beatlocation
        light = spatial_lights[beatlocation][random.randrange(0, 3)]
        beatlocation = spatial_lights_index.index(light)
        if len(combined[i]) == 6:
            if combined[i][3] != 'DEL':
                harmlist.append([combined[i][0] / float(sr_g), light, {'bri': abs(int(combined[i][1])), 'sat': int(combined[i][4]), 'transitiontime': int(combined[i][3] * 1.2), 'hue': int(combined[i][5] * 1.2)}, combined[i][2]])
        if len(combined[i + 1]) == 6:
            if combined[i + 1][3] != 'DEL':
                harmlist.append([combined[i + 1][0] / float(sr_g), light, {'bri': abs(int(combined[i + 1][1] * .3)), 'sat': int(combined[i + 1][4]), 'transitiontime': int(combined[i + 1][3] * .8), 'hue': int(combined[i + 1][5])}, combined[i + 1][2]])
    print('Success!')
    return harmlist

lights_track = percparse(percgraph, beat_times) + arcparse(arcgraph) + harmparse(harmgraph)
lights_track.sort(key=lambda x: x[0])
for i in lights_track:
    i.append('Unmodified')

lock = 'blaps'

for i in range(1, len(lights_track) - 1):
    if lights_track[i][3] == 'arc':
        if lights_track[i + 1][0] < lights_track[i][0] + (float(len(lights)) / 10) + .10 and lights_track[i + 1][4] != 'Safe':
            lights_track[i + 1][4] = 'DEL'
            lights_track[i][4] = 'Safe'
    if lights_track[i][3] == 'low':
        lock = lights_track[i][1]
        if lights_track[i - 1][0] > lights_track[i][0] - .3 and lights_track[i - 1][3] == ('beat' or 'high' or 'low') and lights_track[i - 1][4] != 'Safe':
            lights_track[i - 1][4] = 'DEL'
            lights_track[i][4] = 'Safe'
        if lights_track[i + 1][0] < lights_track[i][0] + .3 and lights_track[i + 1][3] == ('beat' or 'high' or 'low') and lights_track[i + 1][4] != 'Safe':
            lights_track[i + 1][4] = 'DEL'
            lights_track[i][4] = 'Safe'
    if lights_track[i][3] == 'beat':
        if lights_track[i - 1][0] > lights_track[i][0] - .1 and lights_track[i - 1][3] == 'high' and lights_track[i - 1][4] != 'Safe':
            lights_track[i - 1][4] = 'DEL'
            lights_track[i][4] = 'Safe'
        if lights_track[i + 1][0] < lights_track[i][0] + .1 and lights_track[i + 1][3] == 'high' and lights_track[i + 1][4] != 'Safe':
            lights_track[i + 1][4] = 'DEL'
            lights_track[i][4] = 'Safe'
    if lights_track[1][3] == 'low':
        lock = 'blaps'
    if lights_track[i][3] == 'high':
        lights_track[i][1] = lock
        lock = 'blaps'
        if lights_track[i - 1][0] > lights_track[i][0] - .1 and lights_track[i - 1][3] == 'high' and lights_track[i - 1][4] != 'Safe':
            lights_track[i - 1][4] = 'DEL'
            lights_track[i][4] = 'Safe'
        if lights_track[i + 1][0] < lights_track[i][0] + .1 and lights_track[i + 1][3] == 'high' and lights_track[i + 1][4] != 'Safe':
            lights_track[i + 1][4] = 'DEL'
            lights_track[i][4] = 'Safe'

    if type(lights_track[i][1]) == int and lights_track[i][1] == lock and lights_track[i][3] != ('low' or 'high'):
        lights_track[i][4] = 'DEL'
    elif type(lights_track[i][1]) == list:
        if lock in lights_track[i][1] and lights_track[i][3] != ('low' or 'high' or 'beat'):
            lights_track[i][1] = [x for x in lights_track[i][1] if x != lock]

lights_track = [x for x in lights_track if x[4] != 'DEL']

def write_data(filename, graph):
    with open(filename, 'w') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(len(graph)):
            writer.writerow(graph[i])

write_data(os.path.join('E:\\', 'Python_Projects', 'Audio_Engine', 'Logs', 'harmgraph.csv'), harmgraph)
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

def blah():
    execute(os.path.join('E:\\', 'Python_Projects', 'Audio_engine', 'temp.wav'))

blah()
