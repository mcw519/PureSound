# -*- coding: utf-8 -*-
from pytARD_2D.ard import ARDSimulator2D
from pytARD_2D.partition import AirPartition2D
from pytARD_2D.interface import InterfaceData2D, Direction2D

from common.parameters import SimulationParameters
from common.impulse import Unit
from common.animation_plotter import AnimationPlotter, PressureFieldAssembler

import numpy as np

duration = 1.5  #  seconds
Fs = 630  # sample rate
upper_frequency_limit = 60  # Hz
c = 4  # m/s
spatial_samples_per_wave_length = 2

verbose = True
visualize = True

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

partitions.append(AirPartition2D(np.array([
    [room_width],  # X, width
    [room_width]   # Y, depth
]), sim_param, impulse))


partitions.append(AirPartition2D(np.array([
    [room_width],  # X, width
    [room_width]   # Y, depth
]), sim_param))

partitions.append(AirPartition2D(np.array([
    [room_width],  # X, width
    [room_width]   # Y, depth
]), sim_param))


interfaces = []
interfaces.append(InterfaceData2D(0, 1, Direction2D.X)) # Horizontal Interface
interfaces.append(InterfaceData2D(1, 2, Direction2D.Y)) # Vertical Interface

mics = []

sim = ARDSimulator2D(sim_param, partitions, 1, interfaces, mics)
sim.preprocessing()
sim.simulation()

if visualize:
    L = np.array([[1,1],[0,1]])
    pressure_field = PressureFieldAssembler().assemble2d(sim_param, partitions, L)
    
    fps = 30
    anim = AnimationPlotter().plot_2D(pressure_field,
                                      sim_param,
                                      interval= 1000 / fps, # in ms
                                      video_output=True,
                                      file_name='L-Room')  
