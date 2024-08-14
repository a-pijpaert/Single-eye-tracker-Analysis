import numpy as np

def gaze_target(position_target: np.array,
                    position_eye: np.array):

    # Direction vectors from eyes to target
    dir_vec = position_target - position_eye

    # Normalize direction vectors
    dir_normalized = dir_vec / np.linalg.norm(dir_vec)

    # calculate target angles
    phi = np.arcsin(dir_normalized[1])
    theta = np.arcsin(dir_normalized[0] / np.cos(phi))

    phi_degrees = np.degrees(phi)
    theta_degrees = np.degrees(theta)

    return [theta_degrees, phi_degrees]


if __name__ == "__main__":
    gaze_deg = gaze_target(np.array([0,0,40]),
                           np.array([30, 0, 650]))
    print(gaze_deg)

    
