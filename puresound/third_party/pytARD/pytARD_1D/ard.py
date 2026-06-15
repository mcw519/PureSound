from common.parameters import SimulationParameters

from pytARD_1D.interface import Interface1D

import numpy as np
from scipy.fftpack import idct, dct
from tqdm import tqdm
import time


class ARDSimulator1D:
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
        normalization_factor : float
            Normalization multiplier to harmonize amplitudes between partitions.
        interface_data : list, optional
            List of Interface objects. All interfaces of the domain are collected here.
        mics : list, optional
            List of Microphone objects. All microphones placed within the domain are collected here.
        '''

        # Parameter class instance (SimulationParameters)
        self.sim_param: SimulationParameters = sim_param

        # List of partition data (Partition objects)
        self.partitions: list = partitions

        self.interface_data = interface_data
        self.interfaces: Interface1D = None
        if self.interface_data:
            self.interfaces = Interface1D(
                sim_param, partitions, fdtd_acc=self.interface_data[0].fdtd_acc)

        self.normalization_factor = normalization_factor

        self.mics = mics

    def preprocessing(self):
        '''
        Preprocessing stage. Refers to Step 1 in the paper.
        '''

        for i in range(len(self.partitions)):
            self.partitions[i].preprocessing()

    def simulation(self, benchmark_data: list = []):
        '''
        Simulation stage. Refers to Step 2 in the paper.

        Parameters
        ----------
        benchmark_data : List, optional
            Collects execution times and starts benchmarking. Optional.
        '''
        for t_s in tqdm(range(2, self.sim_param.number_of_samples)):
            for interface in self.interface_data:
                # Benchmark handling, for measuring performance between FDTD accuracies
                if self.sim_param.benchmark:
                    start_time = time.time()
                    self.interfaces.handle_interface(interface)
                    benchmark_data.append(time.time() - start_time)
                else:
                    self.interfaces.handle_interface(interface)

            for i in range(len(self.partitions)):
                # Execute DCT for next sample
                self.partitions[i].forces = dct(
                    self.partitions[i].new_forces, n=self.partitions[i].space_divisions, type=2)

                # Updating mode using the update rule in equation 8.
                # Relates to (2 * F^n) / (ω_i ^ 2) * (1 - cos(ω_i * Δ_t)) in equation 8.
                self.partitions[i].force_field = ((2 * self.partitions[i].forces.reshape([self.partitions[i].space_divisions, 1])) / (
                    (self.partitions[i].omega_i) ** 2)) * (1 - np.cos(self.partitions[i].omega_i * self.sim_param.delta_t))

                # Relates to M^(n+1) in equation 8.
                self.partitions[i].M_next = 2 * self.partitions[i].M_current * \
                    np.cos(self.partitions[i].omega_i * self.sim_param.delta_t) - \
                    self.partitions[i].M_previous + \
                    self.partitions[i].force_field

                # Convert modes to pressure values using inverse DCT.
                self.partitions[i].pressure_field = idct(self.partitions[i].M_next.reshape(
                    self.partitions[i].space_divisions), n=self.partitions[i].space_divisions, type=2)

                # Normalize pressure p by using normalization constant.
                self.partitions[i].pressure_field *= self.normalization_factor

                # Add results of IDCT to pressure field
                self.partitions[i].pressure_field_results.append(
                    self.partitions[i].pressure_field.copy())

                # Record signal with mics, if provided
                for m_i in range(len(self.mics)):
                    p_num = self.mics[m_i].partition_number
                    self.mics[m_i].record(self.partitions[p_num].pressure_field[int(
                        self.partitions[p_num].space_divisions * (self.mics[m_i].location / self.partitions[p_num].dimensions))], t_s)

                # Update time stepping to prepare for next time step / loop iteration.
                self.partitions[i].M_previous = self.partitions[i].M_current.copy()
                self.partitions[i].M_current = self.partitions[i].M_next.copy()

                # Update impulses
                self.partitions[i].new_forces = self.partitions[i].impulses[t_s].copy()
