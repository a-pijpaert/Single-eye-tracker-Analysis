import csv
import numpy as np
from typing import Dict, Tuple, List, Union
from helpers.convert_pixels_to_degrees import convert_pixels_to_degrees
from helpers.convert_degrees_2_mm import convert_degrees_2_mm
VIEWING_DISTANCE = 650


class StimulusData:
    """
    A class to import and manage eyetracker data from a CSV file.

    This class reads data from a specified CSV file containing columns for stimulus, start_time, end_time,
    and positionX, positionY. The data is organized into a dictionary where each key is a stimulus
    and its value is another dictionary with start_time, end_time, and positions as keys.

    Attributes:
    - filename (str): Path to the CSV file to load data from.
    - data (Dict[int, Dict[str, np.ndarray]]): Dictionary where each key is a stimulus and its value is 
      another dictionary containing data for start_time, end_time, and positions.

    Usage:
    >>> data_tracker = EyeTrackerData('filename.csv').data
    >>> stimulus_data = data_tracker[1]  # Data for stimulus 1
    >>> start_times = stimulus_data['start_times']
    >>> positions = stimulus_data['positions']
    >>> print(start_times)
    >>> print(positions)

    """
    def __init__(self, filename: str):
        self.filename = filename
        self.data: Dict[int, Dict[str, np.ndarray]] = {}
        self.load_data()

    def load_data(self) -> None:
        with open(self.filename, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                stimulus = int(row["stimulus"])
                if stimulus not in self.data:
                    self.data[stimulus] = {"start_time": [], "end_time": [], "position": [], "position_degrees": [], 'position_mm': []}
                self.data[stimulus]["start_time"].append(float(row["start_time"]))# + 0.5) # offset of 500 ms for start of stimulus (Hosp 2020)
                self.data[stimulus]["end_time"].append(float(row["end_time"]))# - 0.5) # cutoff of last 500 ms before end stimulus
                self.data[stimulus]["position"].append((float(row["positionX"]), float(row["positionY"]))) # voor een scherm van (1920,1080)
                self.data[stimulus]["position_degrees"].append(convert_pixels_to_degrees(self.data[stimulus]["position"]))
                self.data[stimulus]["position_mm"].append(convert_degrees_2_mm(self.data[stimulus]["position_degrees"], VIEWING_DISTANCE))

        # Convert lists to numpy arrays
        for stimulus in self.data:
            for key in self.data[stimulus]:
                self.data[stimulus][key] = np.array(self.data[stimulus][key])


# Usage example
# data_tracker = StimulusData('/home/arthur/Projects/AMESMC/data/kaas/MEASUREMENT/stimulus_timestamps.csv').data
# stimulus_data = data_tracker[1]  # Data for stimulus 1
# start_times = stimulus_data['start_time']
# positions = stimulus_data['position']
# print(start_times)
# print(positions)