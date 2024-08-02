import numpy as np

def vergence_target(position_target: np.array,
                    position_left_eye: np.array,
                    position_right_eye: np.array):
    
    distance_left_eye = np.linalg.norm(position_left_eye - position_target)
    distance_right_eye = np.linalg.norm(position_right_eye - position_target)

    interocular_distance = np.linalg.norm(position_left_eye - position_right_eye)

    vergence_rad = np.arccos((distance_left_eye**2 + distance_right_eye**2 - interocular_distance**2) / (2*distance_left_eye*distance_right_eye))
    vergence_deg = np.degrees(vergence_rad)

    return vergence_deg

if __name__ == "__main__":
    vergence_deg = vergence_target(np.array([0,0,40]),
                                   np.array([30, 0, 650]),
                                   np.array([-30, 0, 650]))
    print(vergence_deg)

    