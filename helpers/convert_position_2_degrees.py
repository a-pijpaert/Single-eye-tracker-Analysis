import numpy as np

VIEWING_DISTANCE = 650

def convert_position_2_degrees(position): # in mm
    T = position # Target position on screen 
    C = np.array([0, 0, VIEWING_DISTANCE]) # 3D position cyclopic eye on the z-axis of the screen

    # Calculate the difference vector S
    S = T - C

    # Normalize S
    S = S / np.linalg.norm(S, axis=1)[:, np.newaxis]

    # Calculate Phi (Vertical angle)
    Phi = np.degrees(np.arcsin(S[:,1]))  # S(2) in MATLAB is S[1] in Python (0-indexed)

    # Calculate Theta (Horizontal angle)
    Theta = np.degrees(np.arcsin(S[:,0] / np.cos(np.radians(Phi))))  # S(1) in MATLAB is S[0] in Python (0-indexed)

    position_degrees = np.column_stack((Theta, Phi, np.zeros(len(Theta))))
    return np.squeeze(position_degrees)