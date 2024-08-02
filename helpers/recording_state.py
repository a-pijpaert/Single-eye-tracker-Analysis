from enum import Enum

class RecordingState(Enum):
    """
    Enum representing different states of a recording process. If the process is IDLE
    data is not saved, if the process is !IDLE, data is saved. 

    The first digit tells you which process is is:
    1 -> IDLE
    2 -> MEASUREMENT
    3 -> CALIBRATION

    The last digit of the value tells you if it is a mono- or binocular measurement:
    1 -> Binocular
    2 -> Left eye
    3 -> Right eye

    Attributes:
        IDLE (int): State when recording is idle. No data is saved
        MEASUREMENT (int): State during the measurement phase.
        CALIBRATION (int): State during the calibration phase.
        CALIBRATION_LEFT (int): Sub-state for left-side calibration within the calibration phase.
        CALIBRATION_RIGHT (int): Sub-state for right-side calibration within the calibration phase.
    """
    IDLE = 11
    IDLE_LEFT = 12
    IDLE_RIGHT = 13

    MEASUREMENT = 21
    MEASUREMENT_LEFT = 22
    MEASUREMENT_RIGHT = 23
    
    CALIBRATION = 31
    CALIBRATION_LEFT = 32
    CALIBRATION_RIGHT = 33

    ORIENTATION = 41

    VERGENCE_PUPIL = 51
    VERGENCE_DISTANCE_1 = 61
    VERGENCE_DISTANCE_2 = 71
    VERGENCE_DISTANCE_3 = 81
