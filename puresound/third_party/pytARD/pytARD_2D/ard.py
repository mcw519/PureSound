from common.parameters import SimulationParameters
from common.notification import Notification

from pytARD_2D.interface import Interface2DStandard, Interface2DLooped

from tqdm import tqdm


class ARDSimulator2D:
    '''
    ARD Simulation class. Creates and runs ARD simulator instance.
    '''

    def __init__(
        self, 
        sim_param: SimulationParameters, 
        partitions: list,
        normalization_factor: float = 1,
        interface_data: list = [],
        mics: list = []
        ):

        '''
        Create and prepare ARD simulator instance.

        Parameters
        ----------
        sim_param : SimulationParameters
            Instance of simulation parameter class.
        partitions : list
            List of Partition objects. All partitions of the domain are collected here.
        normalization_factor : float, optional
            Normalization multiplier to harmonize amplitudes between partitions.
        interface_data : list, optional
            List of Interface objects. All interfaces of the domain are collected here.
        mics : list, optional
            List of Microphone objects. All microphones placed within the domain are collected here.
        '''

        # Parameter class instance (SimulationParameters)
        self.sim_param = sim_param

        # List of partition data (Partition2D objects)
        self.partitions = partitions

        # List of interfaces (InterfaceData objects)
        self.interface_data = interface_data

        if interface_data and interface_data[0].looped:
            self.interfaces = Interface2DLooped(sim_param, partitions)
        else:
            self.interfaces = Interface2DStandard(sim_param, partitions)

        # Initialize & position mics.
        self.mics = mics

        self.normalization_factor = normalization_factor

    def preprocessing(self):
        '''
        Preprocessing stage. Refers to Step 1 in the paper.
        '''

        for i in range(len(self.partitions)):
            self.partitions[i].preprocessing()

    def simulation(self):
        '''
        Simulation stage. Refers to Step 2 in the paper.
        '''
        if self.sim_param.verbose:
            print(f"Simulation started.")
        
        Notification.notify("The ARD Simulation has started. Check terminal for progress end ETA.", "pytARD: Simulation started")

        for t_s in tqdm(range(self.sim_param.number_of_samples)):
            # Global interface handling call
            for interface in self.interface_data:
                self.interfaces.handle_interface(interface)

            # Calling all interfaces to simulate wave propagation locally
            for i in range(len(self.partitions)):
                self.partitions[i].simulate(t_s, self.normalization_factor)

                # Per-partition microphone handling
                for m_i in range(len(self.mics)):
                    p_num = self.mics[m_i].partition_number
                    pressure_field_y = int(self.partitions[p_num].space_divisions_y * (
                        self.mics[m_i].location[1] / self.partitions[p_num].dimensions[1]))
                    pressure_field_x = int(self.partitions[p_num].space_divisions_x * (
                        self.mics[m_i].location[0] / self.partitions[p_num].dimensions[0]))

                    # Recording sound data at given location
                    self.mics[m_i].record(self.partitions[p_num].pressure_field.copy().reshape([
                        self.partitions[p_num].space_divisions_y, 
                        self.partitions[p_num].space_divisions_x, 1]
                    )[pressure_field_y][pressure_field_x], t_s)

        if self.sim_param.verbose:
            print(f"Simulation completed successfully.\n")
        
        Notification.notify("The ARD Simulation has completed successfully.", "pytARD: Simulation completed")

