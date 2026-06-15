from pytARD_3D.ard import ARDSimulator3D
from pytARD_3D.partition import AirPartition3D
from pytARD_3D.interface import InterfaceData3D as Interface
from pytARD_3D.interface import Direction3D as Direction

from common.parameters import SimulationParameters
from common.impulse import Unit

import numpy as np
import time

duration = 1  #  seconds
Fs = 8000  # sample rate
upper_frequency_limit = 200  # Hz
c = 342  # m/s
spatial_samples_per_wave_length = 6
verbose = True

# Compilation of room parameters into parameter class
sim_param = SimulationParameters(
    upper_frequency_limit,
    duration,
    c=c,
    Fs=Fs,
    spatial_samples_per_wave_length=spatial_samples_per_wave_length,
    verbose=verbose,
    visualize=False
)

# Location of impulse that gets emitted into the room.
impulse_location = np.array([[2], [2], [1]])

# Define impulse that gets emitted into the room. Uncomment which kind of impulse you want.
impulse = Unit(sim_param, impulse_location, 10000)

partitions = []
partitions.append(AirPartition3D(np.array([[4.0], [4.0], [2.0]]), sim_param, impulse=impulse))
partitions.append(AirPartition3D(np.array([[4.0], [4.0], [2.0]]), sim_param))
partitions.append(AirPartition3D(np.array([[4.0], [4.0], [2.0]]), sim_param))

times_matrix = []

for i in range(0, 15):
    interfaces = []
    interfaces.append(Interface(0, 1, Direction.X))
    interfaces.append(Interface(1, 2, Direction.Y))

    # Instantiating and executing simulation
    sim = ARDSimulator3D(
        sim_param, 
        partitions, 
        normalization_factor=1, 
        interface_data=interfaces
    )

    start_time = time.time()
    sim.preprocessing()
    sim.simulation()
    end_time = time.time() - start_time

    times_matrix.append(end_time)
    print(f"Matrix multiplication time = {end_time}")

times_looped = []

for i in range(0, 15):
    interfaces = []
    interfaces.append(Interface(0, 1, Direction.X, looped=True))
    interfaces.append(Interface(1, 2, Direction.Y, looped=True))

    # Instantiating and executing simulation
    sim = ARDSimulator3D(
        sim_param, 
        partitions, 
        normalization_factor=1, 
        interface_data=interfaces,
    )
    start_time = time.time()
    sim.preprocessing()
    sim.simulation()
    end_time = time.time() - start_time

    times_looped.append(end_time)
    print(f"Iterative accumulation time = {end_time}")

print("*** RESULTS ***")
print(f"Benchmark results for matrix multiplication = {times_matrix}")
print(f"Benchmark results for iterative accumulation = {times_looped}")
times_matrix = np.array(times_matrix)
times_looped = np.array(times_looped)
print(f"Benchmark average for matrix multiplication = {np.average(times_matrix)}")
print(f"Benchmark average for iterative accumulation = {np.average(times_looped)}")