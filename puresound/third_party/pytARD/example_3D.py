from pytARD_3D.ard import ARDSimulator3D
from pytARD_3D.partition import AirPartition3D, PMLPartition3D, DampingProfile
from pytARD_3D.interface import InterfaceData3D as Interface
from pytARD_3D.interface import Direction3D as Direction


from common.parameters import SimulationParameters
from common.impulse import Gaussian, Unit, WaveFile
from common.serializer import Serializer
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
    [2], # X, width
    [2], # Y, depth
    [2]  # Z, height
])

# Define impulse location that gets emitted into the room
# impulse = Gaussian(sim_param, impulse_location, 1)
impulse = Unit(sim_param, impulse_location, 1, upper_frequency_limit - 1)
# impulse = WaveFile(sim_param, impulse_location, 'clap.wav', 100) # Uncomment for wave file injection

# Damping profile with according Zetta value (how much is absorbed)
dp = DampingProfile(4, c, 1e-3)

partitions = []
# Paritions of the room. Can be 1..n. Add or remove partitions here.
# This example is two air partitions connected by an interface.
# Also, you may provide impulse in the partition(s) of your choosing 
# as the last, optinal parameter.
partitions.append(AirPartition3D(np.array([
    [4], # X, width
    [4], # Y, depth
    [2]  # Z, height
]), sim_param, impulse=impulse))

partitions.append(AirPartition3D(np.array([
    [4], # X, width
    [4], # Y, depth
    [2]  # Z, height
]), sim_param))

# Interfaces of the room. Interfaces connect the room together
interfaces = []
interfaces.append(Interface(0, 1, Direction.X))

# Initialize & position mics.
mics = []
if auralize:
    mics.append(Mic(
        1, [
            int(partitions[0].dimensions[0] / 2), 
            int(partitions[0].dimensions[1] / 2), 
            int(partitions[0].dimensions[2] / 2)
        ], sim_param, 
        # Name of resulting wave file
        f"pytARD_3D_{date.today()}_{datetime.now().time()}"
    ))

# Instantiation serializer for reading and writing simulation state data
serializer = Serializer()

# Instantiating and executing simulation (don't change this)
sim = ARDSimulator3D(
    sim_param, 
    partitions, 
    normalization_factor=1, 
    interface_data=interfaces, 
    mics=mics
)
sim.preprocessing()
sim.simulation()

# Find best peak to normalize mic signal and write mic signal to file
if auralize:
    Mic.write_mic_files(mics, upper_frequency_limit, normalize=True)

# Structure of plot graph. Optional, only for visualization.
plot_structure = [
        # Structure: [Height of domain, width of domain, index of partition to plot on the graph]
        [2, 2, 1],
        [2, 2, 2]
    ]

# Write partitions and state data to disk
if write_to_file:
    if verbose:
        print("Writing state data to disk. Please wait...")
    serializer.dump(sim_param, partitions, mics, plot_structure)

# Plotting waveform
if visualize:
    plotter = Plotter()
    plotter.set_data_from_simulation(sim_param, partitions, mics, plot_structure)
    plotter.plot(enable_colorbar=True)

