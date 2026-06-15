import numpy as np
import enum

from common.parameters import SimulationParameters
from common.finite_differences import FiniteDifferences

class Direction2D(enum.Enum):
    '''
    Direction in which sound waves traverse the interface.
    '''
    X = 'HORIZONTAL'
    Y = 'VERTICAL'
    
class InterfaceData2D():
    '''
    Supporting data structure for interfaces.
    '''
    
    def __init__(
        self, 
        part1_index: int, 
        part2_index: int, 
        direction: Direction2D, 
        fdtd_acc: int = 6,
        looped: bool = False
    ):
        '''
        Creates an instance of interface data between two partitions.

        Parameters
        ----------
        part1_index : int, 
            Index of first partition (index of Partition2D list)
        part2_index : int 
            Index of first partition (index of Partition2D list)
        direction : Direction2D
            Passing direction of the sound wave.
        fdtd_acc : int, optional
            FDTD accuracy.
        looped : bool, optional
            Determines if the interfacing is done iteratively instead of being based on matrix multiplication. Likely is slower.
        '''

        self.part1_index: int = part1_index
        self.part2_index: int = part2_index
        self.direction: Direction2D = direction
        self.fdtd_acc: int = fdtd_acc
        self.looped = looped

class Interface2D():
    '''
    Abstract class for Interface, connecting partitions with each other. Interfaces allow for the passing of sound waves between two partitions.
    '''

    def __init__(
        self, 
        sim_param: SimulationParameters, 
        partitions: list,
        fdtd_order: int = 2,
        fdtd_acc: int = 6
    ):
        '''
        No implementation.

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
        pass
    
    def handle_interface(self, interface_data: InterfaceData2D):
        '''
        No implementation.

        Parameters
        ----------
        interface_data : InterfaceData2D
            InterfaceData instance. Determines which two partitions pass sound waves to each other.
        '''
        pass

class Interface2DStandard(Interface2D):
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

        # 2D FDTD coefficents calculation. Normalize FDTD coefficents with space divisions and speed of sound. 
        fdtd_coeffs_not_normalized = FiniteDifferences.get_laplacian_matrix(fdtd_order, fdtd_acc)
        self.FDTD_COEFFS_X: np.ndarray = fdtd_coeffs_not_normalized * ((sim_param.c / partitions[0].h_x) ** 2)
        self.FDTD_COEFFS_Y: np.ndarray = fdtd_coeffs_not_normalized * ((sim_param.c / partitions[0].h_y) ** 2)

        # FDTD kernel size.
        self.INTERFACE_SIZE: int = int((len(fdtd_coeffs_not_normalized[0])) / 2) 

    def handle_interface(self, interface_data: InterfaceData2D):
        '''
        Handles the travelling of sound waves between two partitions.

        Parameters
        ----------
        interface_data : InterfaceData2D
            InterfaceData instance. Determines which two partitions pass sound waves to each other.
        '''
        if interface_data.direction == Direction2D.X:
            p_left = self.partitions[interface_data.part1_index].pressure_field[:, -self.INTERFACE_SIZE:]
            p_right = self.partitions[interface_data.part2_index].pressure_field[:, :self.INTERFACE_SIZE]

            # Calculate new forces transmitted into room
            pressures_along_y = np.hstack((p_left, p_right))
            new_forces_from_interface_y = np.matmul(pressures_along_y, self.FDTD_COEFFS_Y)

            # Add everything together
            self.partitions[interface_data.part1_index].new_forces[:, -self.INTERFACE_SIZE:] += new_forces_from_interface_y[:, :self.INTERFACE_SIZE]
            self.partitions[interface_data.part2_index].new_forces[:, :self.INTERFACE_SIZE] += new_forces_from_interface_y[:, -self.INTERFACE_SIZE:]

        elif interface_data.direction == Direction2D.Y:
            p_top = self.partitions[interface_data.part1_index].pressure_field[-self.INTERFACE_SIZE:, :]
            p_bot = self.partitions[interface_data.part2_index].pressure_field[:self.INTERFACE_SIZE, :]

            # Calculate new forces transmitted into room
            pressures_along_x = np.vstack((p_top, p_bot))
            new_forces_from_interface_x = np.matmul(self.FDTD_COEFFS_X, pressures_along_x)

            # Add everything together
            self.partitions[interface_data.part1_index].new_forces[-self.INTERFACE_SIZE:, :] += new_forces_from_interface_x[:self.INTERFACE_SIZE, :]
            self.partitions[interface_data.part2_index].new_forces[:self.INTERFACE_SIZE, :] += new_forces_from_interface_x[-self.INTERFACE_SIZE:, :]


class Interface2DLooped(Interface2D):
    '''
    Interface for connecting partitions with each other. Interfaces allow for the passing of sound waves between two partitions.
    Implementation is based on loops and is less efficient than the standard Interface3D.
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

        # 2D FDTD coefficents calculation. Normalize FDTD coefficents with space divisions and speed of sound. 
        fdtd_coeffs_not_normalized: np.ndarray = FiniteDifferences.get_laplacian_matrix(fdtd_order, fdtd_acc)
        self.FDTD_COEFFS_X: np.ndarray = fdtd_coeffs_not_normalized * ((sim_param.c / partitions[0].h_x) ** 2)
        self.FDTD_COEFFS_Y: np.ndarray = fdtd_coeffs_not_normalized * ((sim_param.c / partitions[0].h_y) ** 2)

        # FDTD kernel size.
        self.INTERFACE_SIZE = int((len(fdtd_coeffs_not_normalized[0])) / 2) 

    def handle_interface(self, interface_data: InterfaceData2D):
        '''
        Handles all calculations to enable passing of sound waves between partitions through the interface.

        Parameters
        ----------
        interface_data : InterfaceData2D
            Contains data which two partitions are involved, and in which direction the sound will travel.
        '''
        if interface_data.direction == Direction2D.X:
            for y in range(self.partitions[interface_data.part1_index].space_divisions_y):
                pressure_field_around_interface_y = np.zeros(shape=[2 * self.INTERFACE_SIZE])
                pressure_field_around_interface_y[0 : self.INTERFACE_SIZE] = self.partitions[interface_data.part1_index].pressure_field[y, -self.INTERFACE_SIZE : ].copy()
                pressure_field_around_interface_y[self.INTERFACE_SIZE : 2 * self.INTERFACE_SIZE] = self.partitions[interface_data.part2_index].pressure_field[y, 0 : self.INTERFACE_SIZE].copy()

                # Calculate new forces transmitted into room
                new_forces_from_interface_y = np.matmul(pressure_field_around_interface_y, self.FDTD_COEFFS_Y)

                # Add everything together
                self.partitions[interface_data.part1_index].new_forces[y, -self.INTERFACE_SIZE:] += new_forces_from_interface_y[0:self.INTERFACE_SIZE]
                self.partitions[interface_data.part2_index].new_forces[y, :self.INTERFACE_SIZE] += new_forces_from_interface_y[self.INTERFACE_SIZE : self.INTERFACE_SIZE * 2]
    

        elif interface_data.direction == Direction2D.Y:
            for x in range(self.partitions[interface_data.part1_index].space_divisions_x):
                pressure_field_around_interface_x = np.zeros(shape=[2 * self.INTERFACE_SIZE])
                pressure_field_around_interface_x[0 : self.INTERFACE_SIZE] = self.partitions[interface_data.part1_index].pressure_field[-self.INTERFACE_SIZE : , x].copy()
                pressure_field_around_interface_x[self.INTERFACE_SIZE : 2 * self.INTERFACE_SIZE] = self.partitions[interface_data.part2_index].pressure_field[0 : self.INTERFACE_SIZE, x].copy()

                # Calculate new forces transmitted into room
                new_forces_from_interface_x = np.matmul(pressure_field_around_interface_x, self.FDTD_COEFFS_X)

                # Add everything together
                self.partitions[interface_data.part1_index].new_forces[-self.INTERFACE_SIZE:, x] += new_forces_from_interface_x[0:self.INTERFACE_SIZE]
                self.partitions[interface_data.part2_index].new_forces[:self.INTERFACE_SIZE, x] += new_forces_from_interface_x[self.INTERFACE_SIZE : self.INTERFACE_SIZE * 2]
