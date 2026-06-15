from pytARD_2D.ard import ARDSimulator2D as ARDS
from pytARD_2D.partition import AirPartition2D as PARTD
from pytARD_2D.interface import InterfaceData2D, Direction2D

from common.parameters import SimulationParameters as SIMP
from common.impulse import Gaussian
from common.serializer import Serializer
from common.plotter import Plotter
from common.microphone import Microphone as Mic

from utility_wavdiff import wav_diff, visualize_multiple_waveforms

import numpy as np

'''
Measures sound absorption when a waves are travelling through an interface in 2D.
Control setup is one continuous domain with a single partition, test setup is a seperated domain with
two individual partitions, connected by an interface. Both setups have the same total domain length.
A plot is drawn illustrating the differences between both impulse response files (IRs).
'''

# Room parameters
Fs = 8000  # sample rate
upper_frequency_limit = 300  # Hz
c = 342  # m/s
duration = 0.2  #  seconds
spatial_samples_per_wave_length = 6

verbose = True
visualize = False
write_to_file = True
filename = "verify_2D_interface_absorption"

sim_param = SIMP(
    upper_frequency_limit,
    duration,
    c=c,
    Fs=Fs,
    spatial_samples_per_wave_length=spatial_samples_per_wave_length,
    normalization_constant=1,
    verbose=verbose,
    visualize=visualize
)

impulse_location = np.array([[0.5], [0.5]])
impulse = Gaussian(sim_param, impulse_location, 10000)

control_room = PARTD(np.array([2, 1]), sim_param, impulse)
test_room_left = PARTD(np.array([1, 1]), sim_param, impulse)
test_room_right = PARTD(np.array([1, 1]), sim_param)
control_room = [control_room]
test_room = [test_room_left, test_room_right]

interfaces = []
interfaces.append(InterfaceData2D(1, 0, Direction2D.X))

long_room_mic1 = Mic(
    0,  # Parition number
    # Position
    [0.5, 0.5],
    sim_param,
    filename + "_" + "roomA_mic1"  # Name of resulting wave file
)
long_room_mic2 = Mic(
    0,  # Parition number
    # Position
    [1.5, 0.5],
    sim_param,
    filename + "_" + "roomA_mic2"  # Name of resulting wave file
)

short_room_mic1 = Mic(
    0,
    [0.5, 0.5],
    sim_param,
    filename + "_" + "roomB_mic1"
)
short_room_mic2 = Mic(
    1,
    [0.5, 0.5],
    sim_param,
    filename + "_" + "roomB_mic2"
)

control_mics = [long_room_mic1, long_room_mic2]
test_mics = [short_room_mic1, short_room_mic2]

serializer = Serializer()
plotter = Plotter()

def write_and_plot(room):
    if write_to_file:
        if verbose:
            print("Writing state data to disk. Please wait...")
        serializer.dump(sim_param, room, mics=[],
                        plot_structure=[], filename=filename,)

control_sim = ARDS(sim_param, control_room, .25, mics=control_mics)
control_sim.preprocessing()
control_sim.simulation()

test_sim = ARDS(sim_param, test_room, 1, interfaces, mics=test_mics)
test_sim.preprocessing()
test_sim.simulation()

both_mic_peaks = []
both_mic_peaks.append(Mic.find_peak_multiple_mics(control_mics))
both_mic_peaks.append(Mic.find_peak_multiple_mics(test_mics))
best_peak = np.max(both_mic_peaks)


def write_mic_files(mics, peak):
    for i in range(len(mics)):
        mics[i].write_to_file(peak, upper_frequency_limit)


write_mic_files(control_mics, best_peak)
write_mic_files(test_mics, best_peak)

write_and_plot(test_room)

wav_diff(
    filename + "_" + "roomA_mic1.wav",
    filename + "_" + "roomB_mic1.wav",
    filename + "_" + "mic1_diff.wav"
)
wav_diff(
    filename + "_" + "roomA_mic2.wav",
    filename + "_" + "roomB_mic2.wav",
    filename + "_" + "mic2_diff.wav"
)
visualize_multiple_waveforms([
    filename + "_" + "roomA_mic1.wav",
    filename + "_" + "roomB_mic1.wav",
    filename + "_" + "mic1_diff.wav",
    filename + "_" + "roomA_mic2.wav",
    filename + "_" + "roomB_mic2.wav",
    filename + "_" + "mic2_diff.wav"
], dB=False)
