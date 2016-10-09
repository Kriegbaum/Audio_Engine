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
[Fs, x] = audioBasicIO.readAudioFile(os.path.join('E:\\', 'Python_Projects', 'Audio_engine', 'temp.wav'))
x = audioBasicIO.stereo2mono(x)                                                 # Collapses to mono signal
F = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050*Fs, 0.025*Fs)       #Creates an array of features per frame
print('Extracted!')

N = []                                                                          #Array of features that we are going to use and modify
harmonic_p = []                                                                   #Item for array N, power of higher frequencies
percussive_p = []                                                                 #Item for array N, power of lower frequencies
arc_p = []                                                                        #Item for array N, total track power
spec_center_p = []
chaos_p = []

#PHASE THIS OUT IN FUTURE VERSIONS
perc_colors = [0, 10879, 5461]                                                  #List of hue values for percparse to use

def trim(lst):
    newlst = []
    for i in range(len(lst)):
        if lst[i] == lst[-1]:
            continue
        else:
            newlst.append(float(lst[i]))
    return newlst

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
    harmonic_p.append(tmp)                                                        #Creates harmonic list for N
harmonic = trim(harmonic_p)

for i in range(len(F[1])):                                                      #Combines chroma frequencies and mel frequencies for percussive processing
    tmp = 0
    tmp += F[8][i] * 1.25 * 0
    tmp += F[9][i]
    tmp += F[21][i] * 2
    tmp += F[22][i]
    percussive_p.append(tmp)                                                      #Creates percussive list for N
percussive = trim(percussive_p)

for i in range(len(F[1])):                                                      #Combines all MFCCs for use in arc processing
    tmp = 0
    tmp += F[21][i]
    tmp += F[22][i] * 1.03
    tmp += F[23][i] * 1.06
    tmp += F[24][i] * 1.09
    tmp += F[25][i] * 1.13
    tmp += F[26][i] * 1.16
    tmp += F[27][i] * 1.19
    tmp += F[28][i] * 1.23
    tmp += F[29][i] * 1.26
    tmp += F[30][i] * 1.29
    tmp += F[31][i] * 1.33
    tmp += F[32][i] * 1.36
    tmp2 = 0
    tmp2 += F[9][i]
    tmp2 += F[10][i]
    tmp2 += F[11][i]
    tmp2 += F[12][i]
    tmp2 += F[13][i]
    tmp2 += F[14][i]
    tmp2 += F[15][i]
    tmp2 += F[16][i]
    tmp2 += F[17][i]
    tmp2 += F[18][i]
    tmp2 += F[19][i]
    tmp2 += F[20][i]
    tmp2 += F[21][i]
    arc_p.append(tmp * .75 + tmp2 * .25)                                          #Creates arc list for N
arc = trim(arc_p)

for i in range(len(F[1])):
    tmp = 0
    tmp += F[0][i]
    tmp += F[2][i] * .063
    tmp += F[6][i]
    tmp += F[33][i]
    chaos_p.append(tmp)
chaos = trim(chaos_p)

for i in range(len(F[1])):
    spec_center_p.append(F[3][i])
spec_center = trim(spec_center_p)

N.append(harmonic)                                                              #Wrap all created items into array N
N.append(percussive)
N.append(arc)
N.append(spec_center)
N.append(chaos)

def normalize(sequence, sigma=1):
    values = list(sequence)
    low = min(values)
    high = max(values)
    factor = 255 / (high + abs(low))
    for i in range(len(values)):
        values[i] = factor * (values[i] + abs(low))
    values = gaussian_filter1d(values, sigma)
    newval = []
    for i in range(len(values)):
        newval.append([i, values[i]])
    return newval


def sumgraph(extract, effect=0, sigma=15):                                      #Akin to position/time graph, takes a list of frames, creates a graph
#Effect refers to what item in N is used, sigma is the sigma value for gaussian smoothing
    avg_tmp = 0
    for i in range(len(extract[effect])):                                       #Grab the average power for given N
        avg_tmp += extract[effect][i]
    avg = avg_tmp / len(extract[effect])

    run = avg                                                                   #run is a value that gets modified each frame
    graph = []                                                                  #[x, y] positions for the sumgraph
    #CAN BE DONE WITH LAMBDA, PHASE THIS OUT
    graph_values = []                                                           #Sumgraph values without x value
    for i in range(len(extract[effect])):
        if extract[effect][i] < 0.1 and i != 0:
            extract[effect][i] = abs(avg - 2)
        run += extract[effect][i] - avg                                         #Modifies run up or down depending on how far off average it is
        graph.append([i, 0])
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
        factor = 255 / (deriv_max + deriv_min)
    else:
        factor = 255 / (deriv_max)
    for i in range(len(deriv_graph)):
        if shift == True:
            deriv_graph[i][1] += deriv_min
        deriv_graph[i][1] *= factor
    return deriv_graph

def hue_process_i(value):
    value += random.randrange(-40, 40)
    if value > 255:
        value = 255
    if value < 1:
        value = 1
    if value > 128:
        value = value * 1.32
        if value > 255:
            value = 255
    if value < 128:
        value = value * .55
    value = 255 - value
    hue = abs(value * 200)
    return hue

def hue_process(value):
    value += random.randrange(-10, 10)
    if value > 255:
        value = 255
    if value < 1:
        value = 1
    if value < 1:
        value = 1
    if value > 128:
        value = value * 1.32
        if value > 255:
            value = 255
    if value < 128:
        value = value * .68
    hue = value * 213
    return hue

def write_data(filename, graph):
    with open(filename, 'w') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(len(graph)):
            if type(graph[i]) != list:
                graph[i] = [graph[i]]
                graph[i].append(i)
                graph[i][0], graph[i][1] = graph[i][1], graph[i][0]
            writer.writerow(graph[i])


absarc = normalize(arc, 5)
absharm = normalize(harmonic)
absperc = normalize(percussive)
abschaos = normalize(chaos)

harmgraph = sumgraph(N, 0, 14)
percgraph = sumgraph(N, 1, 1)
arcgraph = sumgraph(N, 2, 30)
arclarge = sumgraph(N, 2, 150)
perclarge = sumgraph(N, 1, 150)

spectral_graph = sumgraph(N, 3, 30)


def write_graphs():
    print('Writing a whole bunch of stuff')
    write_data(os.path.join('E:\\', 'Python_Projects', 'Audio_Engine', 'Logs', 'harmraw.csv'), harmonic)
    write_data(os.path.join('E:\\', 'Python_Projects', 'Audio_Engine', 'Logs', 'arcraw.csv'), arc)
    write_data(os.path.join('E:\\', 'Python_Projects', 'Audio_Engine', 'Logs', 'percraw.csv'), percussive)
    write_data(os.path.join('E:\\', 'Python_Projects', 'Audio_Engine', 'Logs', 'perclarge.csv'), perclarge)
    write_data(os.path.join('E:\\', 'Python_Projects', 'Audio_Engine', 'Logs', 'arclarge.csv'), arclarge)
    write_data(os.path.join('E:\\', 'Python_Projects', 'Audio_Engine', 'Logs', 'arcgraph.csv'), arcgraph)
    write_data(os.path.join('E:\\', 'Python_Projects', 'Audio_Engine', 'Logs', 'harmgraph.csv'), harmgraph)
    write_data(os.path.join('E:\\', 'Python_Projects', 'Audio_Engine', 'Logs', 'percgraph.csv'), percgraph)
    write_data(os.path.join('E:\\', 'Python_Projects', 'Audio_Engine', 'Logs', 'absarc.csv'), absarc)
    write_data(os.path.join('E:\\', 'Python_Projects', 'Audio_Engine', 'Logs', 'absharm.csv'), absharm)
    write_data(os.path.join('E:\\', 'Python_Projects', 'Audio_Engine', 'Logs', 'absperc.csv'), absperc)

write_graphs()
