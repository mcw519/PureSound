from common.parameters import SimulationParameters

from scipy.io.wavfile import write
import numpy as np


class Microphone():
    '''
    Virtual microphone. Can be placed anywhere in the domain, and can create IR (impulse response) files.
    '''

    def __init__(self, partition_number: int, location: np.ndarray, sim_param: SimulationParameters, name: str):
        '''
        Creates and places a virtual microphone inside a partition, depending on the partition number.

        Parameters
        ----------
        partition_number : int
            Number of partition, in which the microphone will be placed.
        location : np.ndarray
            Location of microphone. Relative to chosen partition.
        sim_param : SimulationParameters
            Instance of simulation parameter class.
        name : str
            Name of resulting .WAV file.
        '''
        self.partition_number = partition_number
        self.location = location
        self.sim_param: SimulationParameters = sim_param
        self.signal = np.zeros(sim_param.number_of_samples)
        self.name: str = name + ".wav"

    def record(self, sample: float, index: int):
        '''
        Records the signal sample at given index.

        Parameters
        ----------
        sample : float
            Sampled sound.
        index : int
            Index (time slice) of the sample.
        '''
        self.signal[index] = sample

    def write_to_file(self, normalize_divisor: float = None, lowpass_frequency: float = None):
        '''
        Writes all the collected sound data to a file.

        Parameters
        ----------
        normalize_divisor : float
            Normalization divisor, which helps in getting a unified amplitude before writing the file.
            Makes sure the microphone signal isn't too loud or too quiet.
        lowpass_frequency : float
            Optional lowpass frequency. If desired, a Butterworth lowpass filter is applied before writing the file.
        '''
        # Normalize data
        if normalize_divisor:
            normalized_signal = self.signal / normalize_divisor
        else:
            normalized_signal = self.signal / np.max(self.signal)

        # Butterworth lowpass filtering. Optional.
        if lowpass_frequency:
            from scipy.signal import butter, sosfilt
            sos = butter(10, lowpass_frequency, 'lp',
                         fs=self.sim_param.Fs, output='sos')
            normalized_signal = sosfilt(sos, normalized_signal)

        # Write to file
        write(self.name, self.sim_param.Fs, normalized_signal.astype(np.float))

    @staticmethod
    def find_peak_multiple_mics(mics: list):
        '''
        Finds peak (largest amplitude) of all tracks the microphones recorded in the domain.

        Parameters
        ----------
        mics : list
            List of Microphone objects.

        Returns
        -------
        float
            Biggest peak of all recorded microphone tracks.
        '''

        def find_biggest_amplitude(mics):
            '''
            Inner helper function to find the biggest peak of all mics.

            Parameters
            ----------
            mics : list
                List of Microphone objects.

            Returns
            -------
            float
                Biggest peak of all recorded microphone tracks.
            '''
            peaks = []
            for i in range(len(mics)):
                peaks.append(np.max(mics[i].signal))
            return np.max(peaks)

        all_mic_peaks = []
        all_mic_peaks.append(find_biggest_amplitude(mics))
        return(np.max(all_mic_peaks))

    @staticmethod
    def write_mic_files(mics: list, upper_frequency_limit: int, normalize: bool = False):
        '''
        Writes the sound data of collection of Microphone objects to disk.

        Parameters
        ----------
        mics : list
            List of Microphone objects.
        upper_frequency_limit : int
            Upper frequency limit, maximum frequency of audible spectrum.
        normalize : bool
            If True, automatically normalizes the written sound files to the largest peak the
            program can find in all recorded tracks.
        '''
        peak = 1
        if normalize:
            peak = Microphone.find_peak_multiple_mics(mics)
        for i in range(len(mics)):
            mics[i].write_to_file(peak, upper_frequency_limit)
