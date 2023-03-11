import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

class AudioDescriptor:
        """
        Contains original raw audio data from a track
        As well as various time synchronized processed arrays
        """
        
        def __init__(self, filepath):
                self.filepath = filepath
                print(f'Loading up {filepath}')
                self.raw_track, self.sample_rate = librosa.load(self.filepath)
                self.hop_length = 512
                print('Performing beat analysis...')
                self.tempo, self.beats = librosa.beat.beat_track(y=self.raw_track,
                                                                 sr=self.sample_rate,
                                                                 hop_length=self.hop_length)
                print('Executing harmonic percussive separation...')
                self.harmonic_track,  self.percussive_track = librosa.effects.hpss(self.raw_track)
                
                print('Generating top level arc...')
                self.arc_plot = self.running_average_plot(self.raw_track, name='Arc')

        def running_average_plot(self, plot, name=''):
                """
                Creates a running average plot from a given dataset
                """
                #TODO: Detect and ignore silence at the beginning/end of track
                abs_plot = np.abs(plot)
                mean = np.mean(abs_plot)
                running_avg = abs_plot[0]
                avg_plot = np.zeros(len(abs_plot))
                zero_threshold = np.max(plot) * 0.01
                for i in tqdm(range(len(abs_plot))):
                        running_avg += abs_plot[i] - mean
                        avg_plot[i] = running_avg
                return avg_plot

        def display_plot(self, plot):
                """
                Shows basic line plot for any given array
                """
                plt.plot(plot)
                plt.show()


