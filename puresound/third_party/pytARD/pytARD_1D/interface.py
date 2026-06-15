from common.finite_differences import FiniteDifferences
from common.parameters import SimulationParameters

import numpy as np

class InterfaceData1D():
    '''
    Supporting data structure for interfaces.
    '''

    def __init__(self, part1_index: int, part2_index: int, fdtd_acc: int = 6):
        '''
        Creates an instance of interface data between two partitions.

        Parameters
        ----------
        part1_index : int, 
            Index of first partition (index of Partition1D list)
        part2_index : int 
            Index of first partition (index of Partition1D list)
        fdtd_acc : int, optional
            FDTD accuracy.
        '''
        self.part1_index = part1_index
        self.part2_index = part2_index
        self.fdtd_acc = fdtd_acc


class Interface1D():
    '''
    Interface for connecting partitions with each other. Interfaces allow for the passing of sound waves between two partitions.
    '''

    def __init__(
        self,
        sim_param: SimulationParameters,
        partitions: list,
        fdtd_order: int = 2,
        fdtd_acc: int = 6
    ):
        '''
        Create an Interface for connecting partitions with each other. Interfaces allow for the passing of sound waves between two partitions.

        Parameters
        ----------
        sim_param : SimulationParameters
            Instance of simulation parameter class.
        partitions : list
            List of Partition objects. All partitions of the domain are collected here.
        fdtd_order : int, optional
            FDTD order.
        fdtd_acc : int, optional
            FDTD accuracy.
        '''

        self.partitions = partitions
        self.fdtd_acc = fdtd_acc

        # 1D FDTD coefficents calculation. Normalize FDTD coefficents with space divisions and speed of sound.
        fdtd_coeffs_not_normalized: np.ndarray = FiniteDifferences.get_laplacian_matrix(fdtd_order, fdtd_acc)
        self.FDTD_COEFFS: np.ndarray = fdtd_coeffs_not_normalized * (sim_param.c / partitions[0].h)

        # Interface size derived from FDTD kernel size.
        # If FDTD Accuracy is for example 6, then the interface is 3 on each
        # side of the interface (3 voxels left, 3 voxels right)
        self.INTERFACE_SIZE = int((len(fdtd_coeffs_not_normalized[0])) / 2)

    def handle_interface(self, interface_data: InterfaceData1D):
        '''
        Handles the travelling of sound waves between two partitions.

        Parameters
        ----------
        interface_data : InterfaceData1D
            InterfaceData instance. Determines which two partitions pass sound waves to each other.
        '''
        # Initialize pressure field around interface
        pressure_field_around_interface = np.zeros(
            shape=[2 * self.INTERFACE_SIZE])

        # Left rod
        pressure_field_around_interface[0: self.INTERFACE_SIZE] = self.partitions[
            interface_data.part1_index].pressure_field[-self.INTERFACE_SIZE:].copy()

        # Right rod
        pressure_field_around_interface[self.INTERFACE_SIZE: 2 *
                                        self.INTERFACE_SIZE] = self.partitions[interface_data.part2_index].pressure_field[0: self.INTERFACE_SIZE].copy()

        new_forces_from_interface = np.matmul(
            pressure_field_around_interface, self.FDTD_COEFFS)

        # Add everything together
        self.partitions[interface_data.part1_index].new_forces[-self.INTERFACE_SIZE:
                                                              ] += new_forces_from_interface[0:self.INTERFACE_SIZE]
        self.partitions[interface_data.part2_index].new_forces[:
                                                              self.INTERFACE_SIZE] += new_forces_from_interface[self.INTERFACE_SIZE: self.INTERFACE_SIZE * 2]
