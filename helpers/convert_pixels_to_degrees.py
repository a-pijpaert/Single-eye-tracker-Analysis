import numpy as np

VIEWING_DISTANCE = 650

def convert_pixels_to_degrees(position):
    position_mm = np.array(position) * 0.275 # REF for pixel_pitch: https://www.philips.nl/c-p/243V7QDAB_00/full-hd-lcd-monitor
    position_degrees = np.degrees(np.arctan(position_mm/(VIEWING_DISTANCE)))
    return np.squeeze(position_degrees)
