from common.parameters import SimulationParameters

import string
import numpy as np
from scipy.io.wavfile import read
from scipy.signal import firwin


class Impulse():
    '''
    Abstract impulse class. Has no implementation
    '''

    def __init__(self):
        '''
        This method is deliberately not implemented
        '''
        pass

    def get(self):
        '''
        This method is deliberately not implemented
        '''
        raise NotImplementedError(
            "This method is deliberately not implemented")


class Gaussian(Impulse):
    '''
    Creates and injects a Gaussian impulse as an impulse source.
    '''

    def __init__(
        self,
        sim_param: SimulationParameters,
        location: np.ndarray,
        amplitude: int,
        width: float = 70
    ):
        '''
        Instantiation of a Gaussian impulse.

        Parameters
        ----------
        sim_param : SimulationParameters
            Parameters of ARD Simulation
        location : ndarray
            Location in which the impulse gets injected
        amplitude : int
            Determines the amplitude; "loudness" of impulse
        width : float
            Width of gaussian impulse; determines how "wide" the impulse is, and how long it lasts on peak amplitude.
        '''
        self.sim_param = sim_param
        self.location = location
        self.time_sample_indices = np.arange(
            0, self.sim_param.number_of_samples, 1)
        self.amplitude = amplitude
        self.width = width

    @staticmethod
    def create_gaussian_impulse(x: int, mu: float, sigma: float):
        '''
        Generates a Gaussian impulse as an impulse source.

        Parameters
        ----------
        x : int
            Current signal sample
        mu : float
            Spatial offset.
        sigma : float
            Width of gaussian impulse; determines how "wide" the impulse is, and how long it lasts on peak amplitude.

        Returns
        -------
        float
            Gaussian impulse at signal sample (t(x))
        '''
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sigma, 2.)))

    def get(self):
        '''
        Injects gaussian impulse defined by parameters which were defined on impulse instantiation.

        Returns
        -------
        ndarray
            Impulse over time.
        '''

        impulse = self.amplitude * Gaussian.create_gaussian_impulse(
            self.time_sample_indices, 80 * 4, 80) - self.amplitude * self.create_gaussian_impulse(self.time_sample_indices, 80 * 4 * 2, 80)

        return impulse


class WaveFile(Impulse):
    '''
    Creates and injects a wave file on disk as an impulse source.
    '''

    def __init__(
        self, 
        sim_param: SimulationParameters, 
        location: np.ndarray, 
        path_to_file: string, 
        amplitude: int
    ):
        '''
        Instantiation of a wave file impulse.

        Parameters
        ----------
        sim_param : SimulationParameters
            Parameters of ARD Simulation
        location : ndarray
            Location in which the impulse gets injected
        path_to_file : string
            Location and file name of wave file on disk. Be sure that sample rate matches.
        amplitude : int
            Determines the amplitude; "loudness" of impulse        
        '''
        self.sim_param = sim_param
        self.location = location
        self.amplitude = amplitude

        (wave_fs, self.wav) = read(path_to_file)
        assert(wave_fs == sim_param.Fs), "Wave file sample rate doesn't match simulation sample rate! Adjust Fs accordingly or resample source file."

    def get(self):
        '''
        Injects wave file on disk as impulse source defined by parameters which were defined on impulse instantiation.

        Returns
        -------
        ndarray
            Impulse over time.
        '''

        return self.amplitude * self.wav[0:self.sim_param.number_of_samples]


class Unit(Impulse):
    '''
    Creates and injects a unit impulse as an impulse source.
    '''

    def __init__(
        self, 
        sim_param: SimulationParameters, 
        location: np.ndarray, 
        amplitude: int, 
        cutoff_frequency: int = None, 
        filter_order: int = 41
    ):
        '''
        Instantiation of an unit impulse.

        Parameters
        ----------
        sim_param : SimulationParameters
            Parameters of ARD Simulation
        location : ndarray
            Location (x,y,z) in which the impulse gets injected 
        amplitude : int
            Determines the amplitude; "loudness" of impulse 
        cutoff_frequency : int
            Determines the frequency which will get lowpassed (high frequency are cut off).
        filter_order : int
            Determines the order of the lowpass filter.
        '''
        self.sim_param = sim_param
        self.location = location
        self.amplitude = amplitude
        self.impulse = np.zeros(self.sim_param.number_of_samples)

        if cutoff_frequency is None:
            cutoff_frequency = sim_param.Fs / 2

        self.filter_coeffs = firwin(filter_order, (cutoff_frequency / 2) * 0.95, fs=sim_param.Fs)
        self.impulse[0: len(self.filter_coeffs)] = self.filter_coeffs
        self.impulse[len(self.filter_coeffs): 2 * len(self.filter_coeffs)] = -self.filter_coeffs

    def get(self):
        '''
        Injects unit impulse defined by parameters which were defined on impulse instantiation.

        Returns
        -------
        ndarray
            Impulse over time.
        '''
        return self.amplitude * self.impulse

    def get(self):
        '''
        Injects unit impulse defined by parameters which were defined on impulse instantiation.

        Returns
        -------
        ndarray
            Impulse over time.
        '''

        return self.amplitude * self.impulse
