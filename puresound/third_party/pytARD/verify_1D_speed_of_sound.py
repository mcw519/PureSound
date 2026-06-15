from pytARD_1D.ard import ARDSimulator1D
from pytARD_1D.partition import AirPartition1D

from common.parameters import SimulationParameters
from common.impulse import Unit

import matplotlib.pyplot as plt
import numpy as np

'''
Plots a sound wave at three different times when travelling through an air partition.
Goal is to make sure the speed of sound (c) is accurate.
'''

sim_param = SimulationParameters(
    max_simulation_frequency=500,
    T=1,
    c=342,
    Fs=8000,
    spatial_samples_per_wave_length=6,
    verbose=True,
    visualize=True
)

impulse_location = np.array([[0]])
impulse = Unit(sim_param, impulse_location, 1, cutoff_frequency=500)
c_partition = AirPartition1D(np.array([sim_param.c]), sim_param, impulse)
partitions = [c_partition]

sim = ARDSimulator1D(sim_param, partitions, 1)
sim.preprocessing()
sim.simulation()

if sim_param.visualize:
    room_dims = np.linspace(0., c_partition.dimensions[0], len(
        c_partition.pressure_field_results[0]))
    ytop = np.max(c_partition.pressure_field_results)
    ybtm = np.min(c_partition.pressure_field_results)

    plt.figure()
    sizerino = 18
    plt.rc('font', size=sizerino)  # controls default text size
    plt.rc('axes', titlesize=sizerino)  # fontsize of the title
    plt.rc('axes', labelsize=sizerino)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=sizerino)  # fontsize of the x tick labels
    plt.rc('ytick', labelsize=sizerino)  # fontsize of the y tick labels
    plt.rc('legend', fontsize=sizerino)  # fontsize of the legend

    # T = 0
    plt.subplot(1, 3, 1)
    plt.plot(
        room_dims, c_partition.pressure_field_results[10], 'r', linewidth=3)
    plt.xlabel("Location [m]")
    plt.ylabel("Amplitude")
    plt.ylim(top=ytop)
    plt.ylim(bottom=ybtm)
    plt.vlines(np.min(room_dims), ybtm, ytop, color='gray')
    plt.vlines(np.max(room_dims), ybtm, ytop, color='gray')
    plt.yticks([])
    plt.title(f"T ≈ 0s")
    plt.grid()
    plt.tight_layout()

    # T = 0.5
    plt.subplot(1, 3, 2)
    plt.plot(room_dims, c_partition.pressure_field_results[int(
        len(c_partition.pressure_field_results)/2)], 'r', linewidth=3)
    plt.xlabel("Location [m]")
    plt.ylabel("Amplitude")
    plt.ylim(top=ytop)
    plt.ylim(bottom=ybtm)
    plt.vlines(np.min(room_dims), ybtm, ytop, color='gray')
    plt.vlines(np.max(room_dims), ybtm, ytop, color='gray')
    plt.yticks([])
    plt.title(f"T = 0.5s")
    plt.grid()
    plt.tight_layout()

    # T = 1
    plt.subplot(1, 3, 3)
    plt.plot(
        room_dims, c_partition.pressure_field_results[-10], 'r', linewidth=3)
    plt.xlabel("Location [m]")
    plt.ylabel("Amplitude")
    plt.ylim(top=ytop)
    plt.ylim(bottom=ybtm)
    plt.vlines(np.min(room_dims), ybtm, ytop, color='gray')
    plt.vlines(np.max(room_dims), ybtm, ytop, color='gray')
    plt.yticks([])
    plt.title(f"T ≈ 1s")
    plt.grid()
    plt.tight_layout()

    plt.show()
