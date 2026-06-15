# pytARD
**pytARD** is a free and open source Python room impulse response generator using Adaptive Rectangular Decomposition (ARD) for auralization and visualization of wave distribution and reverberation inside rooms.

![pytARD Logo-modified](https://user-images.githubusercontent.com/61276147/156866082-d0380c16-df85-4a09-a2e0-a3d37db26dc4.jpg)

## Prerequisites
 - **OS:** Developed and tested on GNU/Linux systems (Ubuntu 20.04 LTS), macOS (10.14 and 11.6) and Windows 10
 - **Python:** Python 3 required. Developed and tested on version 3.8.2 and 3.9.2. Older as well as newer versions should run with no problems.
- **Required Python packages**: For core functionality, matplotlib 3.3.4, numpy 1.20.1, scipy 1.6.2 and 4.64.0 needs to be installed.
## Installation
You can use `git clone` to pull this repository on your hard drive, alternative download this repository as a .zip file.
## License
This software is subject to the [AGPL-3.0 license](https://www.gnu.org/licenses/agpl-3.0.en.html). This software comes with no warranty.
## Acknowledgements
If you find this software helpful, feel free to cite us.

> [1] Corrodi, O., Fürbringer & S., Smailov, N.: GPU-based Time Domain Solver for Acoustic Wave Equation.

    @thesis{pytard2022,
        author = {Fürbringer, Severin and Corrodi, Oliver and Smailov, Nikita},
        title = {GPU-based Time Domain Solver for Acoustic Wave Equation},
        school = {ZHAW School of Engineering},
        year = {2022},
        month = {July},
        type = {Bachelor's Thesis}
    }
# Documentation
pytARD comes with three different implementations to simulate 1D, 2D and 3D spaces. For quick reference how to use the software, see `example_1D.py`, `example_2D.py` and `example_3D.py`.
## SimulationParameters
Data container for defining simulation parameters.
```
sim_param = SimulationParameters(
	max_simulation_frequency=250,		# Highest frequency
	T=1,					# Simulation duration
	spatial_samples_per_wave_length=6	# Discrete points per single wave length
	c=343,					# Speed of sound
	Fs=8000,				# Sample rate
	verbose=True				# Console output
	visualize=True				# Visualization of wave distribution
)
```
## Impulses
Impulses can be used as source signal, which is passed off into a certain partition.
### Common parameters
Following parameters are shared between all types of impulse.
 - `amplitude`: How loud the impulse is.
 - `impulse_location`: Position of the impulse. Numpy array with coordinates according to which dimensional implementation of pytARD was used, for example `impulse_location = np.array([[2], [2], [2])` for 3D.

### Impulse types
There are three different types of impulses in pytARD.
 - **Unit impulse:** This is the standard impulse in RIR generators. `impulse = Unit(sim_param, impulse_location, amplitude)`
 - **Gaussian impulse:** `impulse = Gaussian(sim_param, impulse_location, amplitude)`
 - **Wave file:** `impulse = WaveFile(sim_param, impulse_location, 'path_to_file.wav', amplitude)`

## Partitions
Partitions make up parts of the domain (the room to be simulated). Those partitions can be connected via interfaces to allow travel of sound waves from partition to partition.
### Air Partitions
An air partition resembles an empty space in which sound can travel through. Air partitions reflect acoustic waves indefinitely without loss in amplitude.
#### 1D example
```
partition = AirPartition1D(
	4, 				# Partition width
	sim_param, 			# SimulationParameters object
	impulse=impulse			# Impulse object (if impulse is desired)
)
```
#### 2D example
```
air_partition = AirPartition2D(
	np.array([[4.0], [4.0]]), 	# Partition width and height
	sim_param, 			# SimulationParameters object
	impulse=impulse			# Impulse object (if impulse is desired)
)
```
#### 3D example
```
air_partition = AirPartition3D(
	np.array([
		[4], 			# X, width of partition
		[4], 			# Y, depth of partition
		[2] 			# Z, height of partition
	]), 
	sim_param,			# SimulationParameters object
	impulse=impulse			# Impulse object
)
```
### PML Partitions
Perfectly Matched Layer (PML) partitions absorb sound energy depending on the damping profile and its reflection coefficient.
**Important:** Please note that at this time, only 2D PML partitions are working. We're glad to have any kind of contribution.
```
pml_partition = PMLPartition2D(
	np.array([[1.0], [4.0]]), 	# Partition width and height
	sim_param, 		  	# SimulationParameters object
	dp)
)
```
#### Damping Profile
Determines how intense the reflections of the PML partition are, or how much sound energy is absorbed. Be sure to pass the width of the partition as the first parameter.
```
dp = DampingProfile(
	room_width=4, 			# Width of partition
	c=343, 				# Speed of sound
	reflection_coefficient=1e-8	# Reflection coefficient R.
)
```
## Interfaces
Interfaces are used for connecting partitions with each other. Interfaces allow for the passing of sound waves between two partitions. To define interfaces, the helper classes `InterfaceData1D`, `InterfaceData2D` and `InterfaceData3D` is used.
### Example
It is required for all partitions to be collected in a `List` first. The indices of this list is referenced in the interface creation later.
```
domain = [
	air_partition_1, 		# Index 0
	air_partition_2, 		# Index 1
	...		 		# Index n
]
```
An interface is defined by referencing above mentioned List indices. To connect the first two air partition together, their `List` indices are passed as parameters. Since interfaces need a direction (or axis) to travel through, the third parameter is either `Direction3D.X`, `Direction3D.Y` or `Direction3D.Z` for 3D.
```
interface = InterfaceData3D(		# Can also be InterfaceData3D or InterfaceData1D
	0, 				# Origin partition number (partition list index)
	1, 				# Target partition number (partition list index)
	Direction3D.X			# Direction the sound waves travel in. Can also be Direction2D
)
```
## Microphone
To auralize the simulation and create RIRs, virtual microphones are to be created and placed inside a partition within the domain. The `position` parameter needs to be adjusted to the according dimension and position.

Just like [interfaces](##Interfaces), the microphones needs to be mapped to the according partition `List` indices.
```
mic = Mic(
	0, 				# Parition number (partition list index)
	[				# Positioning of microphone:
		1,			# X coordinate of partition (for 1D, 2D and 3D)
		1,			# Y coordinate of partition (for 2D and 3D)
		1			# Z coordinate of partition (for 3D)
	],
	sim_param,
	"RIR"				# Name of resulting wave file
)
```
## ARDSimulator
Room Impulse Responses (RIRs) simulation using the Adaptive Rectangular Decomposition (ARD). For further details see [\[1\]](#Acknowledgements).
```
sim = ARDSimulator3D(			# Can also be ARDSimulator2D or ARDSimulator1D 
	sim_param, 			# SimulationParameters object
	partitions, 			# List of Partition objects
	interface_data=interfaces, 	# List of Interface objects
	mics=mics 			# List of Microphone objects
)
sim.preprocessing()			# Start preprocessing
sim.simulation()			# Start the simulation
```
## Plotter
To visualize the wave distribution, a `Plotter` class is provided. 

To ensure correct display of each partition of the domain, the `plot_structure` variable needs to be adjusted according to following graph:
![99_komplexe_raumformen_plot](https://user-images.githubusercontent.com/61276147/172880366-7030c07c-f857-4e09-ad14-5f6304cb4651.jpg)
Each row represents the partition index incrementally. To ensure correct representation of the setup above, `plot_structure` needs to be configured as following:
```
plot_structure = [
# Structure: [Height of domain, width of domain, index to plot on the graph]
	[2, 3, 1], 			# Partition index 0
	[2, 3, 2], 			# Partition index 1
	[2, 3, 3],			# Partition index 2
	[2, 3, 5]  			# Partition index 3
]
```
To start plotting, use following instructions:
```
plotter = Plotter()
plotter.set_data_from_simulation(sim_param, partitions, mics, plot_structure)
plotter.plot()
```
## Serializer
As the ARD method is resource heavy, a means to save simulation data to disk, as well as post-generation visualization and auralization is provided. Be sure to call serializer after the simulation was completed fully. **Necessary parameters** are `SimulationParameters` and a `List` of `Partition` objects, **optional parameters** are a `List` of `Microphone` objects for auralization and `plot_structure` for visualization.
```
# Instantiation serializer for reading and writing simulation state data
serializer = Serializer()
serializer.dump(sim_param, partitions, mics, plot_structure)
```
