# -*- coding: utf-8 -*-
from pytARD_3D.ard import ARDSimulator3D
from pytARD_3D.partition import AirPartition3D
from pytARD_3D.interface import InterfaceData3D, Direction3D

from common.parameters import SimulationParameters
from common.impulse import Unit
from common.plotter import Plotter
from common.animation_plotter import AnimationPlotter

import numpy as np

duration = 0.5  #  seconds
Fs = 630  # sample rate
upper_frequency_limit = 60  # Hz
c = 4  # m/s
spatial_samples_per_wave_length = 2

verbose = True
visualize = True
use_animation_plotter = False

sim_param = SimulationParameters(
    upper_frequency_limit,
    duration,
    c=c,
    Fs=Fs,
    spatial_samples_per_wave_length=spatial_samples_per_wave_length,
    verbose=verbose,
    visualize=visualize,
    visualize_source=False
)

SCALE = 150  

impulse_location = np.array([
    [int(1)],  # X, width
    [int(1)],  # Y, depth
    [int(1)]  # Z, height
])

impulse = Unit(sim_param, impulse_location, 20, upper_frequency_limit - 1)

room_width = int(2)

partitions = []

partitions.append(AirPartition3D(np.array([
    [room_width],  # X, width
    [room_width],  # Y, depth
    [room_width]  # Z, height
]), sim_param, impulse))


partitions.append(AirPartition3D(np.array([
    [room_width],  # X, width
    [room_width],  # Y, depth
    [room_width]  # Z, height
]), sim_param))

interfaces = []

title = ''
TEST_KIND = ['X', 'Y', 'Z', 'all'][2]
if TEST_KIND == 'X':
    axis = 2
    interfaces.append(InterfaceData3D(0, 1, Direction3D.X))
    title = 'YZ-Inteface'
elif TEST_KIND == 'Y':
    axis = 1
    interfaces.append(InterfaceData3D(0, 1, Direction3D.Y))
    title = 'XZ-Inteface'
elif TEST_KIND == 'Z':
    axis = 0
    interfaces.append(InterfaceData3D(0, 1, Direction3D.Z))
    title = 'XY-Inteface'
elif TEST_KIND == 'all':
    partitions.append(AirPartition3D(np.array([
        [room_width],  # X, width
        [room_width],  # Y, depth
        [room_width]  # Z, height
    ]), sim_param, impulse))

    partitions.append(AirPartition3D(np.array([
        [room_width],  # X, width
        [room_width],  # Y, depth
        [room_width]  # Z, height
    ]), sim_param))

    partitions.append(AirPartition3D(np.array([
        [room_width],  # X, width
        [room_width],  # Y, depth
        [room_width]  # Z, height
    ]), sim_param, impulse))

    partitions.append(AirPartition3D(np.array([
        [room_width],  # X, width
        [room_width],  # Y, depth
        [room_width]  # Z, height
    ]), sim_param))
    
    interfaces.append(InterfaceData3D(0, 1, Direction3D.X))
    interfaces.append(InterfaceData3D(0, 1, Direction3D.X))
    interfaces.append(InterfaceData3D(0, 1, Direction3D.X))
    interfaces.append(InterfaceData3D(0, 1, Direction3D.X))
    interfaces.append(InterfaceData3D(0, 1, Direction3D.X))
    interfaces.append(InterfaceData3D(0, 1, Direction3D.X))

mics = []

sim = ARDSimulator3D(sim_param, partitions, 1, interfaces, mics)
sim.preprocessing()
sim.simulation()

if visualize:
    a = 0
    if use_animation_plotter:
        pfX0 = partitions[0].pressure_field_results
        pfX1 = partitions[1].pressure_field_results
        pf_t = [np.concatenate((pfX0[t], pfX1[t]), axis=axis)
                for t in range(sim_param.number_of_samples)]

        fps = 30
        anim = AnimationPlotter().plot_3D(pf_t,
                                          sim_param,
                                          title,
                                          interval=1000 / fps,  # in ms
                                          source_zyx=partitions[0].src_grid_loc,
                                          direction='z')

    else:
        plotter = Plotter()
        plot_structure = [
            [2, 2, 1],
            [2, 2, 2],
            [2, 2, 3],
            [2, 2, 4]
        ]
        plotter.set_data_from_simulation(sim_param, partitions, plot_structure=plot_structure)
        plotter.plot()
