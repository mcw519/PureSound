from pytARD_2D.partition import Partition2D

from common.notification import Notification
from common.parameters import SimulationParameters

import pickle
import lzma
from datetime import date, datetime
import string

class Serializer():
    '''
    Saves and reads simulation to and from disk.
    '''
    def __init__(self, ):
        '''
        Creates a serializer. No implementation.
        '''
        pass

    def create_filename(self):
        '''
        Generates a file name using time stamps.
        '''
        return f"pytard_{date.today()}_{datetime.now().time()}"

    def dump(self, sim_param: SimulationParameters, partitions: list, mics: list = None, plot_structure: list = None, filename: str = None):
        '''
        Writes simulation state data to disk.

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
        filename : str
            File path to write to.
        '''
        if filename:
            file_path = filename + ".xz"
        else:
            file_path = self.create_filename() + ".xz"
        if sim_param.verbose:
            print(f"Writing state data to disk ({file_path}). Please wait...", end="")
        Notification.notify("Writing state data to disk ({file_path}). Please wait...", "pytARD: Writing state data")
        with lzma.open(file_path, 'wb') as fh:
            pickle.dump((sim_param, partitions, mics, plot_structure), fh)
            fh.close()
        if sim_param.verbose:
            print("Done.")
        Notification.notify("Writing state data completed", "pytARD: Writing state data")


    def read(self, file_path: string):
        '''
        Reads simulation state data from disk.

        Parameters
        ----------
        filename : str
            File path to read from.
        '''
        return pickle.load(lzma.open(file_path, 'rb'))
