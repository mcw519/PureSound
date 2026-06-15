from common.serializer import Serializer

import numpy as np
import matplotlib.pyplot as plt

class Plotter():
    '''
    Plotting class. Caution: The plotter is static and can be a bit of a chore to display everything correctly.
    You have to edit the code for each setup you want to display.
    '''

    def __init__(self, verbose: bool = False):
        '''
        Constructs a new plotter.

        Parameters
        ----------
        verbose : bool
            Prints information on the terminal for debugging and status purposes.
        '''
        self.sim_param = None
        self.partitions = None
        self.verbose = verbose

    def set_data_from_file(self, file_name: str):
        '''
        Injects data from compressed .xz file.
        
        Parameters
        ----------
        file_name : str
            Full path to compressed file.
        '''
        serializer = Serializer()
        (sim_param, partitions, mics, plot_structure) = serializer.read(file_name)
        self.sim_param = sim_param
        self.partitions = partitions
        self.mics = mics
        self.plot_structure = plot_structure

    def set_data_from_simulation(self, sim_param, partitions, mics = None, plot_structure: list = None):
        '''
        Injects data directly from ARD simulation frontend.
        
        Parameters
        ----------
        sim_param : SimulationParameters
            Instance of simulation parameter class.
        partitions : list
            List of Partition objects. All partitions of the domain are collected here.
        mics : list
            List of Microphone objects. All microphones placed within the domain are collected here.
        plot_structure : list
            2D array which correlates partitions to Pyplot subplot numbers (width of domain, height of domain, index of partition). 
            See Pyplot documentation to make sure your plot is displayed correctly.
        '''
        self.sim_param = sim_param
        self.partitions = partitions
        self.mics = mics
        self.plot_structure = plot_structure

    def plot_1D(self):
        room_dims = np.linspace(0., self.partitions[0].dimensions[0], len(self.partitions[0].pressure_field_results[0]))
        ytop = np.max(self.partitions[0].pressure_field_results)
        ybtm = np.min(self.partitions[0].pressure_field_results)

        plt.figure()
        for i in range(0, len(self.partitions[0].pressure_field_results), 50):
            plt.clf()
            plt.title(f"ARD 1D (t = {(self.sim_param.T * (i / self.sim_param.number_of_samples)):.4f}s)")
            plt.subplot(1, 2, 1)
            plt.plot(room_dims, self.partitions[0].pressure_field_results[i], 'r', linewidth=1)
            plt.ylim(top=ytop)
            plt.ylim(bottom=ybtm)
            plt.vlines(np.min(room_dims), ybtm, ytop, color='gray')
            plt.vlines(np.max(room_dims), ybtm, ytop, color='gray')
            plt.subplot(1, 2, 2)
            plt.plot(room_dims, self.partitions[1].pressure_field_results[i], 'b', linewidth=1)
            plt.xlabel("Position [m]")
            plt.ylabel("Displacement")
            plt.ylim(top=ytop)
            plt.ylim(bottom=ybtm)
            plt.vlines(np.min(room_dims), ybtm, ytop, color='gray')
            plt.vlines(np.max(room_dims), ybtm, ytop, color='gray')
            plt.grid()
            plt.pause(0.001)

    def plot_mics(self, axs: list):
        '''
        Plots microphone location on the graph.

        Parameters
        ----------
        axs : list
            List of pyplot axes.
        '''
        for i in range(len(self.mics)):
            # Select current partition
            current_partition = self.mics[i].partition_number

            # Convert meters to indices
            mic_pos_x = int((self.mics[i].location[0] / self.partitions[current_partition].dimensions[0]) * self.partitions[i].space_divisions_x)
            mic_pos_y = int((self.mics[i].location[1] / self.partitions[current_partition].dimensions[1]) * self.partitions[i].space_divisions_y)

            # Plot microphone
            axs[current_partition].plot([mic_pos_x],[mic_pos_y], 'ro', label=f'Microphone {i}')
            axs[i].legend()

    def plot_source(self, axs: list):
        '''
        Plots source location on the graph.

        Parameters
        ----------
        axs : list
            List of pyplot axes.
        '''
        for i in range(len(self.partitions)):
            # Check if impulse source in partition is present
            if self.partitions[i].impulse:
                # Convert meters to indices
                src_pos_x = int((self.partitions[i].impulse.location[0] / self.partitions[i].dimensions[0]) * self.partitions[i].space_divisions_x)
                src_pos_y = int((self.partitions[i].impulse.location[1] / self.partitions[i].dimensions[1]) * self.partitions[i].space_divisions_y)

                # Plot source
                axs[i].plot([src_pos_x],[src_pos_y], 'go', label=f'Source')
                axs[i].legend()

    def check_dimension(self):
        '''
        Returns dimension (1D, 2D or 3D) of domain.

        Returns
        -------
        int
            Number of dimension
        '''
        return len(self.partitions[0].dimensions)

    def plot_2D_3D(self, enable_colorbar: bool = False, speed: int = 50, partition_cutoff: int = None):
        '''
        Plot 2D or 3D (both are supported) domain in real-time.

        Parameters
        ----------
        enable_colorbar : bool
            Displays a color bar, representing amplitude.
        speed : int
            Speed interval to speed up real time plot. The number correlates to number of frames skipped (e.g. 50 equates to only showing each 50th frame).
        partition_cutoff : int
            Partition display cutoff. Limits the number of partitions to be plotted, e.g. 1 just displays the first partition.
        '''
        fig = plt.figure(figsize=plt.figaspect(0.5))
        
        number_of_partitions = len(self.partitions)

        dimension = self.check_dimension()

        # For 3D. Middle of Z axis
        vertical_slice = None
        if dimension == 3:
            vertical_slice = int(self.partitions[0].space_divisions_z / 2)

        # Cutoff to limit plotting to a certain number of partitions
        if partition_cutoff:
            number_of_partitions = partition_cutoff

        axs = []
        for i in range(number_of_partitions):
            axs.append(fig.add_subplot(self.plot_structure[i][0], self.plot_structure[i][1], self.plot_structure[i][2]))

        for i in range(0, len(self.partitions[0].pressure_field_results), speed):
            Z = []

            for partition in self.partitions:
                if dimension == 3:
                    Z.append(partition.pressure_field_results[i][vertical_slice])
                else:
                    Z.append(partition.pressure_field_results[i])

            # Clear axes
            for ax in axs:
                ax.cla()

            plt.title(f"t = {(self.sim_param.T * (i / self.sim_param.number_of_samples)):.4f}s")

            # Plot mics
            if self.mics:
                self.plot_mics(axs)

            # Plot source
            self.plot_source(axs)

            # Plot pressure fields
            image = None
            for i in range(len(axs)):
                image = axs[i].imshow(Z[i])

            # Color bar
            if enable_colorbar:
                cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
                bar = fig.colorbar(image, cax=cbar_ax)

            plt.pause(0.1)

            if enable_colorbar:
                bar.remove()


    def plot(self, enable_colorbar: bool = False, speed: int = 50, partition_cutoff: int = None):
        '''
        Plot 1D, 2D or 3D domain in real-time.

        Parameters
        ----------
        enable_colorbar : bool
            Displays a color bar, representing amplitude.
        speed : int
            Speed interval to speed up real time plot. The number correlates to number of frames skipped (e.g. 50 equates to only showing each 50th frame).
        partition_cutoff : int
            Partition display cutoff. Limits the number of partitions to be plotted, e.g. 1 just displays the first partition.
        '''
        dimension = len(self.partitions[0].dimensions)
        if self.verbose:
            print(f"Data and simulation is {dimension}D.")
        if dimension == 1:
            self.plot_1D()
        if dimension == 2 or dimension == 3:
            self.plot_2D_3D(enable_colorbar, speed, partition_cutoff)

