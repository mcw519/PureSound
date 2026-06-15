from pytARD_3D.ard import ARDSimulator3D
from pytARD_3D.partition import AirPartition3D, PMLPartition3D, DampingProfile
from pytARD_3D.interface import InterfaceData3D, Direction3D

from common.parameters import SimulationParameters
from common.impulse import Unit
from common.plotter import Plotter
from common.microphone import Microphone as Mic

import numpy as np
from datetime import date, datetime


# Room parameters
duration = 1 # seconds
Fs = 8000 # sample rate
upper_frequency_limit = 250 # Hz
c = 342 # m/s
spatial_samples_per_wave_length = 6

# Procedure parameters
verbose = True
auralize= True
visualize = True
write_to_file = False

# Compilation of room parameters into parameter class (don't change this)
sim_param = SimulationParameters(
    upper_frequency_limit, 
    duration, 
    c=c, 
    Fs=Fs,
    spatial_samples_per_wave_length=spatial_samples_per_wave_length, 
    verbose=verbose,
    visualize=visualize
)

# Define impulse location
impulse_location = np.array([
    [1.5], # X, width
    [1.5], # Y, depth
    [1]  # Z, height
])

# Define impulse location that gets emitted into the room
impulse = Unit(sim_param, impulse_location, 1, 249)

partitions = []
# Paritions of the room. Can be 1..n. Add or remove partitions here.
# This example is two air partitions connected by an interface.
# Also, you may provide impulse in the partition(s) of your choosing 
# as the last, optinal parameter.
partitions.append(AirPartition3D(np.array([
    [3], # X, width
    [3], # Y, depth
    [2]  # Z, height
]), sim_param, impulse=impulse))

partitions.append(AirPartition3D(np.array([
    [3], # X, width
    [3], # Y, depth
    [2]  # Z, height
]), sim_param))

partitions.append(AirPartition3D(np.array([
    [3], # X, width
    [3], # Y, depth
    [2]  # Z, height
]), sim_param))

partitions.append(AirPartition3D(np.array([
    [3], # X, width
    [3], # Y, depth
    [2]  # Z, height
]), sim_param))

# Interfaces of the room. Interfaces connect the room together
interfaces = []
interfaces.append(InterfaceData3D(0, 1, Direction3D.X))
interfaces.append(InterfaceData3D(1, 2, Direction3D.X))
interfaces.append(InterfaceData3D(1, 3, Direction3D.Y))

# Initialize & position mics.
mics = []
if auralize:
    mics.append(Mic(
        3, # Index der Partition (von 0 .. n)
        [
            1.5, 
            1.5, 
            1
        ], sim_param, 
        # Name of resulting wave file
        f"pytARD_3D_{date.today()}_{datetime.now().time()}"
    ))

# Instantiating and executing simulation (don't change this)
sim = ARDSimulator3D(sim_param, partitions, 1, interfaces, mics)
sim.preprocessing()
sim.simulation()

# Find best peak to normalize mic signal and write mic signal to file
if auralize:
    Mic.write_mic_files(mics, upper_frequency_limit, normalize=True)

# Structure of plot graph. Optional, only for visualization.
plot_structure = [
    # Structure: [Height of domain, width of domain, index of partition to plot on the graph]
    [2, 3, 1],
    [2, 3, 2],
    [2, 3, 3],
    [2, 3, 5]
]

# Plotting waveform
if visualize:
    plotter = Plotter()
    plotter.set_data_from_simulation(sim_param, partitions, mics, plot_structure)
    plotter.plot(enable_colorbar=True)

