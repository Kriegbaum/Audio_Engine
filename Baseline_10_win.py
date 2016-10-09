from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
import os
import pydub
import numpy as np
import csv
import pyaudio
from threading import Timer
import time
from phue import Bridge
import wave
from scipy.ndimage import gaussian_filter1d
import random
import shutil


bridge = Bridge('10.0.0.10')
lights = [7,8,10,11,12,14]
spatial_lights = [[14, 10, 8], [8, 14, 7], [7, 8, 12], [12, 7, 11], [11, 12, 10], [10, 14, 11]]
spatial_lights_index = [14, 8, 7, 12, 11, 10]


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
[Fs, x] = audioBasicIO.readAudioFile(os.path.join('E:\\', 'Python_Projects', 'Audio_engine', 'temp.wav'))
x = audioBasicIO.stereo2mono(x)                                                 # Collapses to mono signal
F = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050*Fs, 0.025*Fs)       #Creates an array of features per frame
print('Extracted!')

N = []                                                                          #Array of features that we are going to use and modify
harmonic = []                                                                   #Item for array N, power of higher frequencies
percussive = []                                                                 #Item for array N, power of lower frequencies
arc = []                                                                        #Item for array N, total track power
spec_center = []
chaos = []

#PHASE THIS OUT IN FUTURE VERSIONS
perc_colors = [0, 10879, 5461]                                                  #List of hue values for percparse to use

for i in range(len(F[1])):                                                      #Combine top chroma frequencies and entropy for harmonic processing
    tmp = 0
    tmp += F[2][i] * .1
    tmp += F[25][i] * .93
    tmp += F[26][i] * .94
    tmp += F[27][i] * .95
    tmp += F[28][i] * .96
    tmp += F[29][i] * .97
    tmp += F[30][i] * .98
    tmp += F[31][i] * .99
    tmp += F[32][i]
    harmonic.append(tmp)                                                        #Creates harmonic list for N

for i in range(len(F[1])):                                                      #Combines chroma frequencies and mel frequencies for percussive processing
    tmp = 0
    tmp += F[8][i] * 1.25 * 0
    tmp += F[9][i]
    tmp += F[21][i] * 2
    tmp += F[22][i]
    percussive.append(tmp)                                                      #Creates percussive list for N

for i in range(len(F[1])):
    arc.append(F[2][i])

for i in range(len(F[1])):
    tmp = 0
    tmp += F[0][i]
    tmp += F[2][i] * .063
    tmp += F[6][i]
    tmp += F[33][i]
    chaos.append(tmp)

for i in range(len(F[1])):
    spec_center.append(F[3][i])

N.append(harmonic)                                                              #Wrap all created items into array N
N.append(percussive)
N.append(arc)
N.append(spec_center)
N.append(chaos)

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
        if graph_values[i] < .1:
            graph_values[i] = avg

    graph_values = gaussian_filter1d(graph_values, sigma)

    for i in range(len(graph_values)):
        graph[i][1] = graph_values[i]
    return graph

def derive(graph, shift=True):                                                  #Makes a derivative of the supplied sumgraph
#Shift determines whether or not the graph is scaled to 0-255
    deriv_values = []                                                           #Y values
    deriv_graph = []                                                            #X and Y values
    for i in range(len(graph) - 1):                                             #Calculates slope at each frame, skips last frame
        slope = (graph[i + 1][1] - graph[i][1]) / (graph[i + 1][0] - graph[i][0])
        deriv_graph.append([i, slope])
        deriv_values.append(slope)
    deriv_max = max(deriv_values)                                               #Max and min are used for scaling the graph
    deriv_min = abs(min(deriv_values))
    if shift == True:
        minfactor = 128 / deriv_min
        maxfactor = 128 / deriv_max
    else:
        factor = 255 / (deriv_max)
    for i in range(len(deriv_graph)):
        if shift == True:
            if deriv_graph[i][1] >= 0:
                deriv_graph[i][1] *= maxfactor
            if deriv_graph[i][1] < 0:
                deriv_graph[i][1] *= minfactor
            deriv_graph[i][1] += 128
        else:
            deriv_graph[i][1] *= factor
        if deriv_graph[i][1] > 254:
            deriv_graph[i][1] = 254
        if deriv_graph[i][1] < 0:
            deriv_graph[i][1] = 0
    return deriv_graph

def hue_process_i(value):
    if value > 254:
        value = 254
    if value < 10:
        value = 10
    value = 254 - value
    hue = abs(value * 213)
    return hue

def hue_process(value):
    value += random.randrange(-10, 10)
    if value > 255:
        value = 255
    if value < 150:
        value *= .72
    if value < 1:
        value = 1
    hue = value * 213
    return hue

def write_data(filename, graph):
    with open(filename, 'w') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(len(graph)):
                writer.writerow(graph[i])

harmgraph = sumgraph(N, 0, 16)
percgraph = sumgraph(N, 1, 0)
arcgraph = sumgraph(N, 2, 28)
arclarge = sumgraph(N, 2, 150)
perclarge = sumgraph(N, 1, 150)

spectral_graph = sumgraph(N, 3, 30)
spectral = derive(spectral_graph)

harmderiv = derive(harmgraph)
percderiv = derive(percgraph, False)
arcderiv = derive(arcgraph)
largederiv = derive(arclarge)
plargederiv = derive(perclarge)

def arcparse(graph, deriv):
    print('Parsing Arc...')
    zero_list = []
    deriv2 = derive(deriv, False)
    for i in range(len(deriv2) - 1):
        if (deriv2[i][1] < 0 and deriv2[i + 1][1] > 0) or (deriv2[i][1] > 0 and deriv2[i + 1][1] < 0) or deriv2[i][1] == 0:
            zero_list.append(deriv[i])

    for i in range(len(zero_list) - 1):
        gap = zero_list[i + 1][0] - zero_list[i][0]
        if gap < 8:
            gap = 'DEL'
        zero_list[i].append(gap)
    arclist = []
    for i in range(len(zero_list) - 1):
        if len(zero_list[i]) == 3:
            if zero_list[i][2] != 'DEL':
                hue = hue_process_i(largederiv[zero_list[i + 1][0] + 5][1] * .75 + deriv[zero_list[i + 1][0] + 5][1] * .25)
                sat = spectral[zero_list[i + 1][0]][1] * 3 + 200
                if sat > 254:
                    sat = 254
                bri = int(graph[zero_list[i + 1][0]][1] * .5 + zero_list[i + 1][1] * .5)
                if bri > 254:
                    bri = 254
                arclist.append([zero_list[i + 1][0] + 15, lights, {'bri': bri, 'sat': int(sat), 'transitiontime': abs(int(zero_list[i][2]) - 15), 'hue': int(hue)}, 'arc'])
    print('Success!')
    return arclist

beatlocation = 0

def percparse(deriv):
    print('Parsing Beats...')
    zero_list = []
    deriv2 = derive(deriv, False)
    for i in range(len(deriv) - 1):
        if deriv[i][1] < 0 and deriv[i + 1][1] > 0 and abs(deriv2[i][1]) > .80 * np.mean(np.array([x[1] for x in deriv2])):
            if (largederiv[i][1] > 110) or (plargederiv[i][1] > 128):
                zero_list.append(deriv[i])
            else:
                continue
    for i in range(len(zero_list) - 1):
        gap = zero_list[i + 1][0] - zero_list[i][0]
        if gap < 5:
            gap = 'DEL'
        zero_list[i].append(gap)
    perclist = []
    for i in zero_list:
        if len(i) == 3:
            if i[2] != 'DEL':
                sat = 254
                if sat > 254:
                    sat = 254
                bri = int(arcderiv[i[0]][1] * 1.25 + 150) + random.randrange(-75, 22)
                if bri > 254:
                    bri = 254
                hue = int(abs(hue_process(largederiv[i[0]][1] + 65)))
                global beatlocation
                fixture = spatial_lights[beatlocation][random.randrange(0, 3)]
                beatlocation = spatial_lights_index.index(fixture)
                perclist.append([i[0] - 1, fixture, {'bri': bri, 'sat': sat, 'transitiontime': 1, 'hue': hue}, 'beat'])
    print('Dope ass beats in effect!')
    return perclist

harmlocation = 0

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
        if gap < 5:
            gap = 'DEL'
        combined[i].append(gap)
        combined[i].append(graph[combined[i][0]][1])
    for i in range(len(combined) - 1):
        if combined[i][2] == 'low':
            bri = 140 + arcderiv[i][1] * 1.25
            if bri > 254:
                bri = 254
            combined[i][1] = bri
        if len(combined[i]) > 4:
            sat = spectral[i][1] + 128
            if sat > 254:
                sat = 254
            combined[i][4] = sat
            hue = hue_process(harmderiv[i][1])
            combined[i].append(hue)
    harmlist = []
    for i in range(0, len(combined) - 1, 2):
        global beatlocation
        light = spatial_lights[beatlocation][random.randrange(0, 3)]
        beatlocation = spatial_lights_index.index(light)
        if len(combined[i]) == 6:
            if combined[i][3] != 'DEL':
                harmlist.append([combined[i][0], light, {'bri': int(combined[i][1]), 'sat': int(combined[i][4]), 'transitiontime': int(combined[i][3]), 'hue': int(combined[i][5])}, combined[i][2]])
        if len(combined[i + 1]) == 6:
            if combined[i + 1][3] != 'DEL':
                harmlist.append([combined[i + 1][0], light, {'bri': int(combined[i + 1][1] * .3), 'sat': int(combined[i + 1][4]), 'transitiontime': int(combined[i + 1][3] * .55), 'hue': int(combined[i + 1][5])}, combined[i + 1][2]])
    print('Success!')
    return harmlist

def chaosparse(chaos):
    print('Processing Chaos...')
    chaoslist = []
    for i in range(len(chaos)):
        if chaos[i] > .6:
            light = lights[random.randrange(0,6)]
            chaoslist.append([i, light, {'bri': 254, 'sat': 1, 'transitiontime': 0}, 'Chaos'])
            chaoslist.append([i + 5, light, {'bri': 1, 'sat': 1, 'transitiontime': 0}, 'Chaos'])
    print('Ready to burn the world!')
    return chaoslist


lights_track = arcparse(arcgraph, arcderiv) + harmparse(harmgraph, harmderiv) + percparse(percderiv) + chaosparse(N[4])
lights_track.sort(key=lambda x: x[0])
for i in lights_track:
    i.append('Unmodified')

lock = 'blaps'

for i in range(1, len(lights_track) - 1):
    if lights_track[i][3] == 'Chaos':
        if lights_track[i - 1][0] > lights_track[i][0] - 5 and lights_track[i - 1][4] != ('Safe' or 'Chaos'):
            lights_track[i - 1][4] = 'DEL'
            lights_track[i].append('Safe')
        if lights_track[i + 1][0] < lights_track[i][0] + 5 and lights_track[i + 1][4] != ('Safe' or 'Chaos'):
            lights_track[i + 1][4] = 'DEL'
            lights_track[i][4] = 'Safe'
    if lights_track[i][3] == 'arc':
        if lights_track[i - 1][0] > lights_track[i][0] - 23 and lights_track[i - 1][3] != 'Chaos' and lights_track[i - 1][4] != 'Safe':
            lights_track[i - 1][4] = 'DEL'
            lights_track[i][4] = 'Safe'
        if lights_track[i + 1][0] < lights_track[i][0] + 23 and lights_track[i + 1][3] != 'Chaos' and lights_track[i + 1][4] != 'Safe':
            lights_track[i + 1][4] = 'DEL'
            lights_track[i][4] = 'Safe'
    if lights_track[i][3] == 'low':
        lock = lights_track[i][1]
        if lights_track[i - 1][0] > lights_track[i][0] - 5 and lights_track[i - 1][3] == 'beat' and lights_track[i - 1][4] != 'Safe':
            lights_track[i - 1][4] = 'DEL'
            lights_track[i][4] = 'Safe'
        if lights_track[i + 1][0] < lights_track[i][0] + 5 and lights_track[i + 1][3] == 'beat' and lights_track[i + 1][4] != 'Safe':
            lights_track[i + 1][4] = 'DEL'
            lights_track[i][4] = 'Safe'
    if lights_track[i][3] == 'high':
        lock = 'blaps'
        if lights_track[i - 1][0] > lights_track[i][0] - 5 and lights_track[i - 1][3] == 'beat' and lights_track[i - 1][4] != 'Safe':
            lights_track[i - 1][4] = 'DEL'
            lights_track[i][4] = 'Safe'
        if lights_track[i + 1][0] < lights_track[i][0] + 5 and lights_track[i + 1][3] == 'beat' and lights_track[i + 1][4] != 'Safe':
            lights_track[i + 1][4] = 'DEL'
            lights_track[i][4] = 'Safe'

    if type(lights_track[i][1]) == int and lights_track[i][1] == lock and lights_track[i][3] != ('low' or 'high'):
        lights_track[i][4] = 'DEL'
    elif type(lights_track[i][1]) == list:
        if lock in lights_track[i][1] and lights_track[i][3] != ('low' or 'high' or 'beat' or 'Chaos'):
            lights_track[i][1] = [x for x in lights_track[i][1] if x != lock]


lights_track = [x for x in lights_track if x[4] != 'DEL']

def write_graphs():
    print('Writing a whole bunch of stuff')
    write_data(os.path.join('E:\\', 'Python_Projects', 'Audio_Engine', 'Logs', 'arcgraph.csv'), arcgraph)
    write_data(os.path.join('E:\\', 'Python_Projects', 'Audio_Engine', 'Logs', 'arcderiv.csv'), arcderiv)
    write_data(os.path.join('E:\\', 'Python_Projects', 'Audio_Engine', 'Logs', 'harmgraph.csv'), harmgraph)
    write_data(os.path.join('E:\\', 'Python_Projects', 'Audio_Engine', 'Logs', 'percgraph.csv'), percgraph)
    write_data(os.path.join('E:\\', 'Python_Projects', 'Audio_Engine', 'Logs', 'lights_track.csv'), lights_track)
    write_data(os.path.join('E:\\', 'Python_Projects', 'Audio_Engine', 'Logs', 'spectral_graph.csv'), spectral_graph)
    write_data(os.path.join('E:\\', 'Python_Projects', 'Audio_Engine', 'Logs', 'spectral.csv'), spectral)

write_graphs()

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
    execute(os.path.join('E:\\', 'Python_Projects', 'Audio_engine', 'temp.wav'))
bridge.set_light(lights, 'on', True)
blah()
