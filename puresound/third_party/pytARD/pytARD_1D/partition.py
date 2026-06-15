from common.parameters import SimulationParameters
from common.impulse import Impulse

import numpy as np

class AirPartition1D:
    '''
    Air partition. Resembles an empty space in which sound can travel through.
    '''
    def __init__(
        self,
        dimensions: np.ndarray,
        sim_param: SimulationParameters,
        impulse: Impulse = None
    ):
        '''
        Creates an air partition

        Parameters
        ----------
        dimensions : ndarray
            Size of the partition (room) in meters. 
        sim_param : SimulationParameters
            Instance of simulation parameter class.
        impulse : Impulse, optional
            If an Impulse object is passed, the according impulse is generated on this partition. 
        '''
        self.dimensions = dimensions
        self.sim_param: SimulationParameters = sim_param

        # Voxel grid spacing. Changes automatically according to frequency
        self.h: float = AirPartition1D.calculate_h(self.sim_param)
        AirPartition1D.check_CFL(self.sim_param, self.h)

        # Longest room dimension length dividied by H (voxel grid spacing).
        self.space_divisions = int(dimensions[0] / self.h)

        # Instantiate forces array, which corresponds to F in update rule (results of DCT computation).
        self.forces: np.ndarray = None

        # Instantiate updated forces array. Combination of impulse and/or contribution of the interface.
        # DCT of new_forces will be written into forces.
        self.new_forces: np.ndarray = None

        # Impulse array which keeps track of impulses in space over time.
        self.impulses: np.ndarray = np.zeros(
            shape=[self.sim_param.number_of_samples, self.space_divisions])

        # Array, which stores air pressure at each given point in time in the voxelized grid
        self.pressure_field = None

        # Array for pressure field results (auralisation and visualisation)
        self.pressure_field_results: list = []

        # Fill impulse array with impulses.
        if impulse:
            # Emit impulse into room
            self.impulses[:, int(
                self.space_divisions * (impulse.location[0] / dimensions[0]))] = impulse.get()

        if sim_param.verbose:
            print(
                f"Created partition with dimensions {self.dimensions} m\n ℎ: {self.h} | Space divisions: {self.space_divisions} ({self.dimensions/self.space_divisions} m each)")

    def preprocessing(self):
        '''
        Preprocessing stage. Refers to Step 1 in the paper. Precomputation of DCTs, instantiation of omega_i, and instantiation of time stepping.
        '''
        # Preparing pressure field. Equates to function p(x) on the paper.
        self.pressure_field = np.zeros(
            shape=[self.space_divisions])

        # Precomputation for the DCTs to be performed. Transforming impulse to spatial forces. Skipped partitions as of now.
        self.new_forces = self.impulses[0].copy()

        # Relates to equation 5 and 8 of "An efficient GPU-based time domain solver for the
        # acoustic wave equation" paper.
        # For reference, see https://www.microsoft.com/en-us/research/wp-content/uploads/2016/10/4.pdf.
        self.omega_i: np.ndarray = self.sim_param.c * np.pi * \
            (np.arange(0, self.space_divisions, 1) / np.max(self.dimensions))

        # Workaround. Without this, the calculation of update rule (equation 9) would crash.
        self.omega_i[0] = 0.1

        # Convert omega_i from row vector to column vector
        self.omega_i = self.omega_i.reshape([len(self.omega_i), 1])

        # Update time stepping. Relates to M^(n+1) and M^n in equation 8.
        self.M_previous = np.zeros(shape=[self.space_divisions, 1])
        self.M_current = np.zeros(shape=self.M_previous.shape)
        self.M_next = None

        if self.sim_param.verbose:
            print(f"Shape of omega_i: {self.omega_i.shape}")
            print(f"Shape of pressure field: {self.pressure_field.shape}")

    @staticmethod
    def check_CFL(sim_param, h):
        '''
        Checks stability of wave equation via calculating Courant-Friedrichs-Lewy number.

        Parameters
        ----------
        sim_param : SimulationParameters
            Instance of simulation parameter class.
        h : float
            Voxel length (grid spacing)
        '''
        CFL = sim_param.c * sim_param.delta_t * (1 / h)
        CFL_target = np.sqrt(1/3)
        assert(CFL <= CFL_target), f"Courant-Friedrichs-Lewy number (CFL = {CFL}) is greater than {CFL_target}. \
            Wave equation is unstable. Try using a higher sample rate or more spatial samples per wave length."
        if sim_param.verbose:
            print(f"CFL = {CFL}")

    @staticmethod
    def calculate_h(sim_param: SimulationParameters):
        '''
        Calculates voxel grid spacing ℎ.

        Parameters
        ----------
        sim_param : SimulationParameters
            Instance of simulation parameter class.

        Returns
        -------
        float
            Voxel grid spacing (ℎ).
        '''
        return SimulationParameters.calculate_voxelization_step(sim_param)
