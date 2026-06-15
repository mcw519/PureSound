from common.impulse import Gaussian
from common.parameters import SimulationParameters as SIMP
from common.microphone import Microphone as Mic

from pytARD_1D.ard import ARDSimulator1D as ARDS
from pytARD_1D.partition import AirPartition1D as PARTD
from pytARD_1D.interface import InterfaceData1D

import numpy as np
import csv

'''
Measures average calculation time for each FDTD accuracy (4, 6 and 10).
For each accuracy, a csv file is generated with seperate iteration time results.
'''

src_pos = [0] # m
duration = 2 # seconds
Fs = 8000 # sample rate
upper_frequency_limit = 300 # Hz
c = 342 # m/s
spatial_samples_per_wave_length = 6

auralize = False
verbose = False
visualize = True
filename = "verify_1D_interface_reflection"

sim_param = SIMP(
    upper_frequency_limit, 
    duration, 
    c=c, 
    Fs=Fs,
    spatial_samples_per_wave_length=spatial_samples_per_wave_length, 
    verbose=verbose,
    visualize=False,
    benchmark=True
)

impulse = Gaussian(sim_param, [0], 10000)

result_time_4 = []
result_time_6 = []
result_time_10 = []

for accuracy in [4, 6, 10]:
    for i in range(10):
        test_partition_1 = PARTD(np.array([c]), sim_param, impulse)
        test_partition_2 = PARTD(np.array([c]), sim_param)
        test_room = [test_partition_1, test_partition_2]

        interfaces = []
        interfaces.append(InterfaceData1D(0, 1, fdtd_acc=accuracy))

        mic = Mic(0, int(c / 2), sim_param, filename + "_" +"mic")
        test_mics = [mic]

        test_sim = ARDS(sim_param, test_room, 1, interface_data=interfaces, mics=test_mics)
        test_sim.preprocessing()
        if accuracy == 4:
            test_sim.simulation(result_time_4)
        elif accuracy == 6:
            test_sim.simulation(result_time_6)
        elif accuracy == 10:
            test_sim.simulation(result_time_10)

def write_csv(filename, data):
    with open(filename, 'w') as csvfile: 
        csvwriter = csv.writer(csvfile) 
        for i in range(len(data)):
            csvwriter.writerow([data[i]]) 


for accuracy in [4, 6, 10]:
    if accuracy == 4:
        write_csv(f"verify_1D_fdtd_accuracy_times_{accuracy}.csv", result_time_4)
    elif accuracy == 6:
        write_csv(f"verify_1D_fdtd_accuracy_times_{accuracy}.csv", result_time_6)
    elif accuracy == 10:
        write_csv(f"verify_1D_fdtd_accuracy_times_{accuracy}.csv", result_time_10)


print(f"Average time of accuracy = 4: {np.sum(result_time_4) / len(result_time_4)}")
print(f"Average time of accuracy = 6: {np.sum(result_time_6) / len(result_time_6)}")
print(f"Average time of accuracy = 10: {np.sum(result_time_10) / len(result_time_10)}")

