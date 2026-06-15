from common.impulse import Gaussian
from common.parameters import SimulationParameters as SIMP
from common.microphone import Microphone as Mic

from pytARD_1D.ard import ARDSimulator1D as ARDS
from pytARD_1D.partition import AirPartition1D as PARTD
from pytARD_1D.interface import InterfaceData1D

from utility_wavdiff import wav_diff, visualize_multiple_waveforms 

import numpy as np

'''
Measures sound absorption when a waves are travelling through an interface in 1D.
Control setup is one continuous domain with a single partition, test setup is a seperated domain with
two individual partitions, connected by an interface. Both setups have the same total domain length.
A plot is drawn illustrating the differences between both impulse response files (IRs).
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
filename = "verify_1D_interface_absorption"

# Compilation of room parameters into parameter class
sim_param = SIMP(
    upper_frequency_limit, 
    duration, 
    c=c, 
    Fs=Fs,
    spatial_samples_per_wave_length=spatial_samples_per_wave_length, 
    verbose=verbose,
    visualize=visualize
)

impulse = Gaussian(sim_param, [0], 10000)

# Define test and control rooms
control_partition = PARTD(np.array([c * 2]), sim_param, impulse)
control_room = [control_partition]

test_partition_1 = PARTD(np.array([c]), sim_param, impulse)
test_partition_2 = PARTD(np.array([c]), sim_param)
test_room = [test_partition_1, test_partition_2]

# Define Interfaces
interfaces = []
interfaces.append(InterfaceData1D(0, 1, fdtd_acc=6))

# Define and position mics

# Initialize & position mics. 
control_mic_before = Mic(0, int(c / 2), sim_param, filename + "_" + "roomA_mic1")
control_mic_after = Mic(0, int(3 * (c / 2)), sim_param, filename + "_" +"roomA_mic2")
control_mics = [control_mic_before, control_mic_after]

test_mic_before = Mic(0, int(c / 2), sim_param, filename + "_" +"roomB_mic1")
test_mic_after = Mic(1, int(c / 2), sim_param, filename + "_" +"roomB_mic2")
test_mics = [test_mic_before, test_mic_after]

# Instantiating and executing simulation
control_sim = ARDS(sim_param, control_room, normalization_factor=.5, mics=control_mics)
control_sim.preprocessing()
control_sim.simulation()

test_sim = ARDS(sim_param, test_room, normalization_factor=1, interface_data=interfaces, mics=test_mics)
test_sim.preprocessing()
test_sim.simulation()

# Normalizing + writing recorded mic tracks
def find_best_peak(mics):
    peaks = []
    for i in range(len(mics)):
        peaks.append(np.max(mics[i].signal))
    return np.max(peaks)

both_mic_peaks = []
both_mic_peaks.append(find_best_peak(control_mics))
both_mic_peaks.append(find_best_peak(test_mics))
best_peak = np.max(both_mic_peaks)

def write_mic_files(mics, peak=1):
    for i in range(len(mics)):
        mics[i].write_to_file(peak, upper_frequency_limit)

write_mic_files(control_mics, best_peak)
write_mic_files(test_mics, best_peak)

wav_diff(filename + "_" +"roomA_mic1.wav", filename + "_" +"roomB_mic1.wav", filename + "_" +"mic1_diff.wav")
wav_diff(filename + "_" +"roomA_mic2.wav", filename + "_" +"roomB_mic2.wav", filename + "_" +"mic2_diff.wav")
visualize_multiple_waveforms([filename + "_" +"roomA_mic1.wav", filename + "_" +"roomB_mic1.wav", filename + "_" +"mic1_diff.wav", filename + "_" +"roomA_mic2.wav", filename + "_" +"roomB_mic2.wav", filename + "_" +"mic2_diff.wav"], dB=True)
