from common.impulse import Gaussian
from common.parameters import SimulationParameters as SIMP
from common.microphone import Microphone as Mic

from pytARD_1D.ard import ARDSimulator1D as ARDS
from pytARD_1D.partition import AirPartition1D as PARTD
from pytARD_1D.interface import InterfaceData1D

from scipy.io.wavfile import read

import matplotlib.pyplot as plt
import numpy as np

'''
Test script to measure amplitude of interface reflection.
By ARD definition, all interfaces create a reflection smaller than -40 dB when sound is passing through. 
A plot is drawn to highlight amplitude differences of reflections when different FDTD accuracies are chosen.
'''

# Room parameters
src_pos = [0] # m
duration = 2 # seconds
Fs = 8000 # sample rate
upper_frequency_limit = 300 # Hz
c = 342 # m/s
spatial_samples_per_wave_length = 6

# Procedure parameters
auralize = False
verbose = True
visualize = True
filename = "verify_1D_interface_reflection"

# Compilation of room parameters into parameter class
sim_param = SIMP(
    upper_frequency_limit, 
    duration, 
    c=c, 
    Fs=Fs,
    spatial_samples_per_wave_length=spatial_samples_per_wave_length, 
    verbose=verbose,
    visualize=False
)

impulse = Gaussian(sim_param, [0], 10000)

for accuracy in [4, 6, 10]:
    test_partition_1 = PARTD(np.array([c]), sim_param, impulse)
    test_partition_2 = PARTD(np.array([c]), sim_param)
    test_room = [test_partition_1, test_partition_2]
    interfaces = []
    interfaces.append(InterfaceData1D(0, 1, fdtd_acc=accuracy))

    mic = Mic(0, int(c / 2), sim_param, filename + "_" +"mic")
    test_mics = [mic]

    test_sim = ARDS(sim_param, test_room, 1, interface_data=interfaces, mics=test_mics)
    test_sim.preprocessing()
    test_sim.simulation()

    def find_best_peak(mics):
        peaks = []
        for i in range(len(mics)):
            peaks.append(np.max(mics[i].signal))
        return np.max(peaks)

    both_mic_peaks = []
    both_mic_peaks.append(find_best_peak(test_mics))
    best_peak = np.max(both_mic_peaks)

    def write_mic_files(mics, peak=1):
        for i in range(len(mics)):
            mics[i].write_to_file(peak, upper_frequency_limit)

    write_mic_files(test_mics, best_peak)

    def visualize_ripple(path):
        fs, w = read(path)

        y = np.zeros(shape=len(w))
        t = np.linspace(0, len(w) / fs, len(w))

        for j in range(len(y)):
            if w[j] != 0:
                y[j] = 20 * np.log10(abs(w[j]))

        sizerino = 18
        plt.rc('font', size=sizerino) #controls default text size
        plt.rc('axes', titlesize=sizerino) #fontsize of the title
        plt.rc('axes', labelsize=sizerino) #fontsize of the x and y labels
        plt.rc('xtick', labelsize=sizerino) #fontsize of the x tick labels
        plt.rc('ytick', labelsize=sizerino) #fontsize of the y tick labels
        plt.rc('legend', fontsize=sizerino) #fontsize of the legend
        plt.plot(t, y, label=f"FDTD accuracy = {accuracy}")
        plt.xlabel('Time s')
        plt.ylabel('Amplitude dBA')
        plt.ylim(top=-45, bottom=-47.5)
        plt.xlim(left=1.53, right=1.6)
        print(f"Peak at FDTD accuracy = {accuracy}: {np.max(y[int(1.53*Fs) : int(1.6*Fs)])}")
        plt.grid()

    visualize_ripple(filename + "_" +"mic.wav")

plt.legend()
plt.show()

