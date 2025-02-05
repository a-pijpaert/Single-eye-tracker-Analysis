#%%
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns

# %matplotlib inline

from helpers.recording_state import RecordingState
from helpers.vergence_target import vergence_target
from helpers.gaze_target import gaze_target
from helpers.stimulus_data import StimulusData

# set subject IDs
subject_ID = 's008'
font_size = 8
figures_dir = 'figures'

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
    target_gaze_left = np.empty((len(time),2))
    target_gaze_left[:] = np.nan
    target_gaze_right = np.empty((len(time),2))
    target_gaze_right[:] = np.nan
    for index, (key, value) in enumerate(stimulus_timestamps.items()):
        stimulus_position = stimulus_positions[index]
        start_time = value['start_time'] - timestamp0
        end_time = value['end_time'] - timestamp0
        indices = [i for i, t in enumerate(time) if start_time <= t <= end_time]
        target_vergence[indices] = vergence_target(stimulus_position, left_cor, right_cor)
        target_gaze_left[indices,:] = gaze_target(stimulus_position, left_cor)
        target_gaze_right[indices,:] = gaze_target(stimulus_position, right_cor)


    left_ver = np.arcsin(left_visual_axis[:,1])
    left_hor = np.arcsin(left_visual_axis[:,0] / np.cos(left_ver))
    left_ver_deg = np.degrees(left_ver)
    left_hor_deg = np.degrees(left_hor)

    right_ver = np.arcsin(right_visual_axis[:,1])
    right_hor = np.arcsin(right_visual_axis[:,0] / np.cos(right_ver))
    right_ver_deg = np.degrees(right_ver)
    right_hor_deg = np.degrees(right_hor)

    gaze_vergence = np.degrees(np.arccos(np.sum(left_visual_axis * right_visual_axis, axis=1))) # np.sum(var1 * var2, axis=1) takes the elementwise dotproduct for each row of both vars

    axs[0,index_procedure].plot(time, left_hor_deg, color='#4c72b0', label='Left Eye')
    axs[0,index_procedure].plot(time, right_hor_deg, color='#dd8452',label='Right Eye')
    axs[0,index_procedure].plot(time, target_gaze_left[:,0], color='black', label='Target Left Eye')
    axs[0,index_procedure].plot(time, target_gaze_right[:,0], color='black', linestyle=':', label='Target Right Eye')
    axs[0,index_procedure].set_ylim([-20,20])
    axs[1,index_procedure].plot(time, left_ver_deg, color='#4c72b0')
    axs[1,index_procedure].plot(time, right_ver_deg, color='#dd8452')
    axs[1,index_procedure].plot(time, target_gaze_left[:,1], color='black')
    axs[1,index_procedure].plot(time, target_gaze_right[:,1], color='black', linestyle=':')
    axs[1,index_procedure].set_ylim([-20,20])
    axs[2,index_procedure].plot(time, gaze_vergence, color='#55a868', label='Measured')
    axs[2,index_procedure].plot(time, target_vergence, color='black', label='Target')
    axs[2,index_procedure].set_ylim([2,11])

# axs[0,0].set_title('Targets at 350 mm from subject')
# axs[0,1].set_title('Targets at 610 mm from subject')
# axs[0,0].set_ylabel(u'Horizontal POG\n(\N{DEGREE SIGN})')
# axs[1,0].set_ylabel(u'Vertical POG\n(\N{DEGREE SIGN})')
axs[0,0].set_ylabel(u'Horizontal\nVisual Axis\nAngle (\N{DEGREE SIGN})')
axs[1,0].set_ylabel(u'Vertical\nVisual Axis\nAngle (\N{DEGREE SIGN})')
axs[2,0].set_xlabel('Time (s)')
axs[2,1].set_xlabel('Time (s)')
axs[2,0].set_ylabel(u'Vergence (\N{DEGREE SIGN})')
# axs[2,0].set_ylim([4,7])
# axs[2,1].set_ylim([7,10])
axs[0,1].legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=font_size)
axs[2,1].legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=font_size)

# add labels to the subplots
axs[0, 0].annotate('A', xy=(0.01, 0.99), xycoords='axes fraction', fontsize=font_size, ha='left', va='top')
axs[0, 1].annotate('B', xy=(0.01, 0.99), xycoords='axes fraction', fontsize=font_size, ha='left', va='top')
axs[1, 0].annotate('C', xy=(0.01, 0.99), xycoords='axes fraction', fontsize=font_size, ha='left', va='top')
axs[1, 1].annotate('D', xy=(0.01, 0.99), xycoords='axes fraction', fontsize=font_size, ha='left', va='top')
axs[2, 0].annotate('E', xy=(0.01, 0.99), xycoords='axes fraction', fontsize=font_size, ha='left', va='top')
axs[2, 1].annotate('F', xy=(0.01, 0.99), xycoords='axes fraction', fontsize=font_size, ha='left', va='top')

figure.tight_layout()
rcParams.update({'font.size': font_size})
plt.show()


figure.savefig(f'{figures_dir}/single subject vergence trace.png', bbox_inches='tight')
# %%
