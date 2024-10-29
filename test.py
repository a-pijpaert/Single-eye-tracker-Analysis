import numpy as np

loaded_data = np.load(f'data/s001/analysis data/data_vergence_pupil.npz')

for key in loaded_data.keys():
    print(key)

print(loaded_data)