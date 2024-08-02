#%%
import os
import numpy as np
import matplotlib.pyplot as plt

# %matplotlib inline

from helpers.recording_state import RecordingState
from helpers.vergence_target import vergence_target
from helpers.stimulus_data import StimulusData

# set subject IDs
subject_ID = 's008'
# set parameters
procedures = [RecordingState.VERGENCE_DISTANCE_3.name.lower(),
              RecordingState.VERGENCE_DISTANCE_1.name.lower()]
stimulus_positions_z = [300, 40]
figure, axs = plt.subplots(3,2)


for index_procedure, procedure in enumerate(procedures):
    # load data
    loaded_data = np.load(f'data/{subject_ID}/analysis data/data_{procedure}.npz')
    left_visual_axis = loaded_data['left_visual_axis']
    right_visual_axis = loaded_data['right_visual_axis']
    left_pog_degrees = loaded_data['left_pog_degrees']
    right_pog_degrees = loaded_data['right_pog_degrees']
    left_cor = loaded_data['left_cor']
    right_cor = loaded_data['right_cor']
    stimulus_number = loaded_data['stimulus_number']
    time = loaded_data['time']
    timestamp0 = loaded_data['timestamps'][0]

    stimulus_timestamps = StimulusData(f'/home/arthur/Projects/AMESMC/data/{subject_ID}/{procedure.upper()}/stimulus_timestamps.csv').data
    stimulus_positions = np.array([[70, -70, stimulus_positions_z[index_procedure]],
                                [0, 0, stimulus_positions_z[index_procedure]],
                                [-70, 70, stimulus_positions_z[index_procedure]],
                                [-70, -70, stimulus_positions_z[index_procedure]],
                                [70, 70, stimulus_positions_z[index_procedure]],
                                ])

    target_vergence = np.empty((len(time)))
    target_vergence[:] = np.nan
    for index, (key, value) in enumerate(stimulus_timestamps.items()):
        stimulus_position = stimulus_positions[index]
        start_time = value['start_time'] - timestamp0
        end_time = value['end_time'] - timestamp0
        indices = [i for i, t in enumerate(time) if start_time <= t <= end_time]
        target_vergence[indices] = vergence_target(stimulus_position, left_cor, right_cor)

    left_ver = np.arcsin(left_visual_axis[:,1])
    left_hor = np.arcsin(left_visual_axis[:,0] / np.cos(left_ver))
    left_ver_deg = np.degrees(left_ver)
    left_hor_deg = np.degrees(left_hor)

    right_ver = np.arcsin(right_visual_axis[:,1])
    right_hor = np.arcsin(right_visual_axis[:,0] / np.cos(right_ver))
    right_ver_deg = np.degrees(right_ver)
    right_hor_deg = np.degrees(right_hor)

    gaze_vergence = np.degrees(np.arccos(np.sum(left_visual_axis * right_visual_axis, axis=1))) # np.sum(var1 * var2, axis=1) takes the elementwise dotproduct for each row of both vars
    # gaze_vergence = np.abs(left_hor_deg - right_hor_deg) # enkel horizontal vergentie?
    # axs[0,index_procedure].plot(time, left_pog_degrees[:,0], label='OS')
    # axs[0,index_procedure].plot(time, right_pog_degrees[:,0], label='OD')
    # axs[0,index_procedure].set_ylim([-15,15])
    # axs[1,index_procedure].plot(time, left_pog_degrees[:,1])
    # axs[1,index_procedure].plot(time, right_pog_degrees[:,1])
    # axs[1,index_procedure].set_ylim([-15,15])

    axs[0,index_procedure].plot(time, left_hor_deg, label='OS')
    axs[0,index_procedure].plot(time, right_hor_deg, label='OD')
    axs[0,index_procedure].set_ylim([-20,20])
    axs[1,index_procedure].plot(time, left_ver_deg)
    axs[1,index_procedure].plot(time, right_ver_deg)
    axs[1,index_procedure].set_ylim([-20,20])
    axs[2,index_procedure].plot(time, gaze_vergence, color='g', label='Measured')
    axs[2,index_procedure].plot(time, target_vergence, color='black', label='Target')
    axs[2,index_procedure].set_ylim([2,11])

axs[0,0].set_title('Targets at 350 mm from subject')
axs[0,1].set_title('Targets at 610 mm from subject')
# axs[0,0].set_ylabel(u'Horizontal POG\n(\N{DEGREE SIGN})')
# axs[1,0].set_ylabel(u'Vertical POG\n(\N{DEGREE SIGN})')
axs[0,0].set_ylabel(u'Horizontal\nVisual Axis\nAngle (\N{DEGREE SIGN})')
axs[1,0].set_ylabel(u'Vertical\nVisual Axis\nAngle (\N{DEGREE SIGN})')
axs[2,0].set_xlabel('Time (s)')
axs[2,1].set_xlabel('Time (s)')
axs[2,0].set_ylabel(u'Vergence (\N{DEGREE SIGN})')
# axs[2,0].set_ylim([4,7])
# axs[2,1].set_ylim([7,10])
axs[0,1].legend(loc='upper right')
axs[2,1].legend(loc='upper right')
figure.tight_layout()
plt.show()
# %%
