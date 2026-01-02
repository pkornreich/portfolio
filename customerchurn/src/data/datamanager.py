# BS"D

from pathlib import Path

class DataManager():

    @staticmethod
    def is_running_from_src():
        cwd = Path.cwd()
        # Check if the current working directory ends with a 'src' segment
        return cwd.name == 'src'

