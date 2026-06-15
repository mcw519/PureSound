

class SimulationParameters:
    def __init__(
        self,
        max_simulation_frequency: float,
        T: int,
        spatial_samples_per_wave_length: int = 6,
        c: int = 343,
        Fs: int = 8000,
        normalization_constant: float = 1,
        auralize: bool = None,
        verbose: bool = False,
        visualize: bool = False,
        visualize_source: bool = True,
        benchmark: bool = False
    ):
        '''
        Parameter container class for ARD simulation. Contains all relevant data to instantiate
        and run ARD simulator.

        Parameters
        ----------
        partitions : ndarray
            Collection of 1D, 2D or 3D partitions (rooms). Array of Partition objects.
        src_pos : ndarray
            Location of signal source inside the room. Can be 1D, 2D or 3D.
        max_simulation_frequency : float
            Uppermost frequency of simulation. Can be dialed in lower to enhance performance.
        T : float
            Simulation time [s].
        spatial_samples_per_wave_length : int, optional
            Number of spatial samples per wave length. Usually 2 to 6. Lower values decrease 
            resolution but enhances performance.
        c : float, optional
            Speed of sound [m/s]. Depends on air temperature, pressure and humidity. 
        Fs : int, optional
            Sampling rate. The higher, the more fidelity and higher maximum frequency but at the expense of performance.
        normalization_constant : float, optional
            Normalization multiplier to equalize amplitude across entire domain.
        auralize : ndarray, optional
            Auralizes the room (= makes hearable) by creating an impulse response (IR).
            Format is a list with mic positions. If microphone array is empty, no auralization is being made.
        verbose : bool, optional
            Prints information on the terminal for debugging and status purposes.
        visualize : bool, optional
            Visualizes wave propagation in the plot.
        visualize_source : bool, optional
            Visualizes impulse source in the plot.
        benchmark : bool, optional
            Enables performance benchmarking for comparing different accuracies.
        '''

        assert(T > 0), "Error: Simulation duration must be a positive number greater than 0."
        assert(Fs > 0), "Error: Sample rate must be a positive number."
        assert(c > 0), "Error: Speed of sound must be a positive number."
        assert(max_simulation_frequency > 0), "Error: Uppermost frequency of simulation must be a positive number."
        assert(max_simulation_frequency < (Fs / 2)), "Nyquist-Shannon theorem violated. Make sure upper frequency limit is Fs / 2."

        self.max_simulation_frequency = max_simulation_frequency
        self.c = c
        self.Fs = Fs
        self.T = T
        self.spatial_samples_per_wave_length = spatial_samples_per_wave_length

        # Calculating the number of samples the simulation takes.
        self.number_of_samples = int(T * Fs)

        # Calculate time stepping (Δ_t)
        self.delta_t = T / self.number_of_samples

        self.normalization_constant = normalization_constant
        self.auralize = auralize
        self.verbose = verbose
        self.visualize = visualize
        self.visualize_source = visualize_source
        self.benchmark = benchmark

        if verbose:
            print(
                f"Insantiated simulation.\nNumber of samples: {self.number_of_samples} | Δ_t: {self.delta_t}")

    @staticmethod
    def calculate_voxelization_step(sim_param):
        '''
        Calculate voxelization step for the segmentation of the room (voxelizing the scene).
        The cell size is fixed by the chosen wave length and the number of spatial samples per 
        wave length.
        Calculation: ℎ <= 𝑐 / (2 * 𝑓_max)

        Parameters
        ----------
        sim_param : SimulationParameters
            ARD simulation parameter object

        Returns
        -------
        float
            ℎ, the voxelization step. In numerics and papers, it's usually referred to ℎ. 
        '''
        return sim_param.c / (sim_param.spatial_samples_per_wave_length * sim_param.max_simulation_frequency)
