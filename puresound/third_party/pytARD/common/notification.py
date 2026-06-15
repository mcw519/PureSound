import os
import platform


class Notification:
    '''
    Notifies the user of action the software is currently doing via desktop notification. Supports Linux and macOS.
    '''

    def __init__(self):
        '''
        No implementation
        '''
        pass

    @staticmethod
    def notify(text: str, title: str):
        '''
        Notifies the user of action the software is currently doing via desktop notification.

        Parameters
        ----------
        title : str
            Title of the notification
        message : str
            Message of the notification

        '''
        plt = platform.system()
        if plt == 'Darwin':
            command = f'''
            osascript -e 'display notification "{text}" with title "{title}"'
            '''
        if plt == 'Linux':
            command = f'''
            notify-send "{text}" "{title}"
            '''
        else:
            return
        os.system(command)
