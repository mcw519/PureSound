from pytARD_2D.ard import ARDSimulator2D
from pytARD_2D.partition import AirPartition2D, PMLPartition2D, DampingProfile
from pytARD_2D.interface import InterfaceData2D as Interface
from pytARD_2D.interface import Direction2D as Direction


from common.parameters import SimulationParameters
from common.impulse import Gaussian, Unit, WaveFile
from common.serializer import Serializer
from common.plotter import Plotter
from common.microphone import Microphone as Mic

import numpy as np
from datetime import date, datetime


# Simulation parameters
duration = 1  #  seconds
Fs = 8000  # sample rate
upper_frequency_limit = 200  # Hz
c = 342  # m/s
spatial_samples_per_wave_length = 6

# Procedure parameters
verbose = True
auralize = True
visualize = True
write_to_file = False

# Compilation of room parameters into parameter class
sim_param = SimulationParameters(
    upper_frequency_limit,
    duration,
    c=c,
    Fs=Fs,
    spatial_samples_per_wave_length=spatial_samples_per_wave_length,
    verbose=verbose,
    visualize=visualize
)

# Location of impulse that gets emitted into the room.
impulse_location = np.array([[2], [2]])

# Define impulse that gets emitted into the room. Uncomment which kind of impulse you want.
#impulse = Gaussian(sim_param, impulse_location, 10000)
impulse = Unit(sim_param, impulse_location, 10000)
#impulse = WaveFile(sim_param, impulse_location, 'clap_48000.wav', 1000)

# Damping profile with according Zetta value (how much is absorbed)
dp = DampingProfile(4, c, 1e-8)

partitions = []
# Paritions of the room. Can be 1..n. Add or remove partitions here.
# This example is a square room surrounded by absorbing PML partitions.
# Also, you may provide impulse in the partition(s) of your choosing.
partitions.append(AirPartition2D(np.array([[4.0], [4.0]]), sim_param, impulse=impulse))
partitions.append(PMLPartition2D(np.array([[1.0], [4.0]]), sim_param, dp))
partitions.append(PMLPartition2D(np.array([[1.0], [4.0]]), sim_param, dp))
partitions.append(PMLPartition2D(np.array([[4.0], [1.0]]), sim_param, dp))
partitions.append(PMLPartition2D(np.array([[4.0], [1.0]]), sim_param, dp))

# Interfaces of the room. Interfaces connect the partitions together
interfaces = []
interfaces.append(Interface(1, 0, Direction.X))
interfaces.append(Interface(2, 0, Direction.X))
interfaces.append(Interface(3, 0, Direction.Y))
interfaces.append(Interface(4, 0, Direction.Y))

# Microphones. Add and remove microphones here by copying or deleting mic objects.
# Only gets used if the auralization option is enabled.
mics = []
if auralize:    
    mics.append(Mic(
        0,  # Parition number
        # Position
        [
            1,
            1
        ],
        sim_param,
        # Name of resulting wave file
        f"pytARD_2D_{date.today()}_{datetime.now().time()}"
    ))

# Instantiation serializer for reading and writing simulation state data
serializer = Serializer()

# Instantiating and executing simulation
sim = ARDSimulator2D(
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
    # Structure: [Height of domain, width of domain, index of partition to plot on the graph (min: 1, max: width*height)]
    [3, 3, 5],
    [3, 3, 4],
    [3, 3, 6],
    [3, 3, 2],
    [3, 3, 8]
]

# Write partitions and state data to disk
if write_to_file:
    serializer.dump(sim_param, partitions, mics, plot_structure)

# Plotting waveform
if visualize:
    plotter = Plotter()
    plotter.set_data_from_simulation(sim_param, partitions, mics, plot_structure)
    plotter.plot(enable_colorbar=True)
