from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
import numpy as np
import csv
import pyaudio
from threading import Timer
import time
from phue import Bridge
import wave
from scipy.ndimage import gaussian_filter1d
import random


bridge = Bridge('10.0.0.10')
lights = [7,8,10,11,12,14]

def sample(i):
    command = lights_track[i][2]
    bridge.set_light(lights_track[i][1], command)

[Fs, x] = audioBasicIO.readAudioFile('/Users/andykauff/Python_Code/Audio_Engine/elgar2.wav')
x = audioBasicIO.stereo2mono(x)
F = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050*Fs, 0.025*Fs)


N = []
harmonic = []
percussive = []
arc = []

perc_colors = [0, 10879, 5461]

for i in range(len(F[1])):
    tmp = 0
    tmp += F[3][i] * .15
    tmp += F[26][i] * .93
    tmp += F[27][i] * .94
    tmp += F[28][i] * .95
    tmp += F[29][i] * .96
    tmp += F[30][i] * .97
    tmp += F[31][i] * .98
    tmp += F[32][i] * .99
    tmp += F[33][i]
    harmonic.append(tmp)

for i in range(len(F[1])):
    tmp = 0
    tmp += F[9][i] * 1.25
    tmp += F[10][i]
    tmp += F[22][i] * 1.25
    tmp += F[23][i]
    percussive.append(tmp)

for i in range(len(F[1])):
    arc.append(F[2][i])

N.append(harmonic)
N.append(percussive)
N.append(arc)


def sumgraph(extract, effect=0, sigma=15):
    avg_tmp = 0
    for i in range(len(extract[effect])):
        avg_tmp += extract[effect][i]
    avg = avg_tmp / len(extract[effect])

    run = avg
    graph = []
    graph_values = []
    for i in range(len(extract[effect])):
        run += extract[effect][i] - avg
        graph.append([i, 0])
        graph_values.append(run)

    graph_min = abs(min(graph_values))
    graph_max = max(graph_values)
    factor = 255 / (graph_max + graph_min)

    for i in range(len(graph)):
        graph_values[i] += graph_min
        graph_values[i] *= factor

    graph_values = gaussian_filter1d(graph_values, sigma)

    for i in range(len(graph_values)):
        graph[i][1] = graph_values[i]

    return graph


def derive(graph, shift=True):
    deriv_values = []
    deriv_graph = []
    for i in range(len(graph) - 1):
        slope = (graph[i + 1][1] - graph[i][1]) / (graph[i + 1][0] - graph[i][0])
        deriv_graph.append([i, slope])
        deriv_values.append(slope)
    deriv_max = max(deriv_values)
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

def write_data(filename, graph):
    with open(filename, 'w') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(len(graph)):
                writer.writerow(graph[i])

harmgraph = sumgraph(N, 0, 15)
percgraph = sumgraph(N, 1, 1)
arcgraph = sumgraph(N, 2, 30)

harmderiv = derive(harmgraph)
percderiv = derive(percgraph, False)
arcderiv = derive(arcgraph)

def arcparse(graph, deriv):
    print('Parsing Arc...')
    zero_list = []
    deriv2 = derive(deriv, False)
    for i in range(len(deriv2) - 1):
        if (deriv2[i][1] < 0 and deriv2[i + 1][1] > 0) or (deriv2[i][1] > 0 and deriv2[i + 1][1] < 0) or deriv2[i][1] == 0:
            zero_list.append(deriv[i])

    for i in range(len(zero_list) - 1):
        zero_list[i][1] = zero_list[i + 1][1]
        gap = zero_list[i + 1][0] - zero_list[i][0]
        if gap < 8:
            gap = 'DEL'
        zero_list[i].append(gap)
        zero_list[i].append(graph[zero_list[i][0]][1])
    arclist = []
    for i in zero_list:
        if len(i) == 4:
            if i[2] != 'DEL':
                arclist.append([i[0], lights, {'bri': int(i[1]), 'sat': int(i[3]), 'transitiontime': int(i[2]), 'hue': 54399}, 'arc'])
    print('Success!')
    return arclist

def percparse(deriv):
    print('Parsing Beats...')
    zero_list = []
    deriv2 = derive(deriv, False)
    for i in range(len(deriv) - 1):
        if deriv[i][1] < 0 and deriv[i + 1][1] > 0 and abs(deriv2[i][1]) > np.mean(np.array([x[1] for x in deriv2])):
            if arcgraph[i][1] < 100 and (arcgraph[i + 1][1] - arcgraph[i][1]) < 0:
                continue
            else:
                zero_list.append(deriv[i])
    for i in range(len(zero_list) - 1):
        gap = zero_list[i + 1][0] - zero_list[i][0]
        if gap < 5:
            gap = 'DEL'
        zero_list[i].append(gap)
    perclist = []
    for i in zero_list:
        if len(i) == 3:
            if i[2] != 'DEL':
                sat = int(arcgraph[i[0]][1] * 1.75)
                if sat > 254:
                    sat = 254
                bri = int(arcderiv[i[0]][1] * 1.5)
                if bri > 254:
                    bri = 254
                perclist.append([i[0] - 1, lights[random.randrange(0,6)], {'bri': bri, 'sat': sat, 'transitiontime': 1, 'hue': perc_colors[random.randrange(0,2)]}, 'beat'])
    print('Success!')
    return perclist

def harmparse(graph, deriv):
    print('Parsing Harmonies...')
    low = []
    high = []
    combined = []
    deriv2 = derive(deriv, False)
    for i in range(len(deriv2) - 1):
        if (deriv2[i][1] < 0 and deriv2[i + 1][1] > 0):
            low.append(deriv[i] + ['low'])
        elif (deriv2[i][1] > 0 and deriv2[i + 1][1] < 0) or deriv2[i][1] == 0:
            high.append(deriv[i] + ['high'])
    for i in range(min([len(low), len(high)])):
        combined.append(low.pop(0))
        if high[0][0] > combined[-1][0]:
            combined.append(high.pop(0))
        elif high[1][0] > combined[-1][0]:
            combined.append(high.pop(1))
            del high[0]
    for i in range(len(combined) - 1):
        combined[i][1] = combined[i + 1][1]
        gap = combined[i + 1][0] - combined[i][0]
        if gap < 5:
            gap = 'DEL'
        combined[i].append(gap)
        combined[i].append(graph[combined[i][0]][1])
    harmlist = []
    for i in range(0, len(combined) - 1, 2):
        light = lights[random.randrange(0,6)]
        if len(combined[i]) == 5:
            if combined[i][3] != 'DEL':
                harmlist.append([combined[i][0], light, {'bri': int(combined[i][1]), 'sat': int(combined[i][4]), 'transitiontime': int(combined[i][3]), 'hue': 43519}, combined[i][2]])
        if len(combined[i + 1]) == 5:
            if combined[i + 1][3] != 'DEL':
                harmlist.append([combined[i + 1][0], light, {'bri': int(combined[i + 1][1]), 'sat': int(combined[i + 1][4]), 'transitiontime': int(combined[i + 1][3]), 'hue': 43519}, combined[i + 1][2]])
    print('Success!')
    return harmlist

harmlist = harmparse(harmgraph, harmderiv)
lights_track = harmparse(harmgraph, harmderiv) + arcparse(arcgraph, arcderiv)
lights_track.sort(key=lambda x: x[0])

lock = 'blaps'

for i in range(1, len(lights_track) - 1):
    if lights_track[i][3] == 'arc':
        if lights_track[i - 1][0] > lights_track[i][0] - 5:
            lights_track[i - 1].append('DEL')
        if lights_track[i + 1][0] < lights_track[i][0] + 5:
            lights_track[i + 1].append('DEL')
    if lights_track[i][3] == 'low':
        lock = lights_track[i][1]
        if lights_track[i - 1][0] > lights_track[i][0] - 5 and lights_track[i - 1][3] == 'beat':
            lights_track[i - 1].append('DEL')
        if lights_track[i + 1][0] < lights_track[i][0] + 5 and lights_track[i + 1][3] == 'beat':
            lights_track[i + 1].append('DEL')
    if lights_track[i][3] == 'high':
        lock = 'blaps'
        if lights_track[i - 1][0] > lights_track[i][0] - 5 and lights_track[i - 1][3] == 'beat':
            lights_track[i - 1].append('DEL')
        if lights_track[i + 1][0] < lights_track[i][0] + 5 and lights_track[i + 1][3] == 'beat':
            lights_track[i + 1].append('DEL')

    if type(lights_track[i][1]) == int and lights_track[i][1] == lock and lights_track[i][3] != ('low' or 'high'):
        lights_track[i].append('DEL')
    elif type(lights_track[i][1]) == list:
        if lock in lights_track[i][1] and lights_track[i][3] != ('low' or 'high'):
            lights_track[i][1] = [x for x in lights_track[i][1] if x != lock]


lights_track = [x for x in lights_track if len(x) < 5]


write_data('/Users/andykauff/Desktop/arcgraph.txt', arcgraph)
write_data('/Users/andykauff/Desktop/harmgraph.txt', harmgraph)
write_data('/Users/andykauff/Desktop/percgraph.txt', percgraph)
write_data('/Users/andykauff/Desktop/harms.txt', harmlist)
write_data('/Users/andykauff/Desktop/lights_track.txt', lights_track)


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
        Timer(lights_track[i][0] / 40, sample, [i]).start()

    time.sleep(0.1)

    while data != '':
        stream.write(data)
        data = wf.readframes(chunk)

    stream.close()
    p.terminate()

def blah():
    execute('/Users/andykauff/Python_Code/Audio_Engine/elgar2.wav')
bridge.set_light(lights, 'on', True)
blah()
