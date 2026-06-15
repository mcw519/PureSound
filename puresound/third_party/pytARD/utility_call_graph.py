from pytARD_2D.ard import ARDSimulator2D
from pytARD_2D.partition import AirPartition2D, PMLPartition2D, DampingProfile
from pytARD_2D.interface import InterfaceData2D, Direction2D

from common.parameters import SimulationParameters
from common.impulse import Unit

import numpy as np
from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput
from pycallgraph import GlobbingFilter
from pycallgraph import Config

'''
Utility script to create call graphs with PyCallGraph.
'''
c = 342
sim_param = SimulationParameters(
    200,
    1,
    c=c,
    Fs=8000,
    spatial_samples_per_wave_length=6,
    verbose=False,
    visualize=False
)

partitions = [
    AirPartition2D(np.array([[4.0], [4.0]]), sim_param, impulse=Unit(sim_param, np.array([[2], [2]]), 1, 100)),
    PMLPartition2D(np.array([[1.0], [4.0]]), sim_param, DampingProfile(4, c, 1e-8))
]

interfaces = [
    InterfaceData2D(1, 0, Direction2D.X)
]

# Instantiating and executing simulation
config = Config()
config.trace_filter = GlobbingFilter(exclude=[
    'tqdm.*', '*.tqdm', 'tqdm', 'pycallgraph.*', '*.secret_function', 'multiprocessing', '*.multiprocessing', 'multiprocessing.*'
])

with PyCallGraph(output=GraphvizOutput(), config=config):
    sim = ARDSimulator2D(sim_param, partitions, 1, interfaces)
    sim.preprocessing()
    sim.simulation()
