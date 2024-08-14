#%%
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import statsmodels.formula.api as smf
import seaborn as sns
from __plot_params import *
# %matplotlib qt


from helpers.recording_state import RecordingState
from helpers.vergence_target import vergence_target

# set subject IDs
subject_IDs = ['s001',
               's002',
               's003',
               's007',
               's008',]

# set plotting params
plt.rcParams['font.size'] = font_size

# set parameters
procedures = [RecordingState.VERGENCE_DISTANCE_3.name.lower(),
              RecordingState.VERGENCE_DISTANCE_2.name.lower(),
              RecordingState.VERGENCE_DISTANCE_1.name.lower(),]
stimulus_positions_z = [300, 170, 40]

# figure, axs = plt.subplots(3,2)
counter = 0
MAEs_vergence = {}
measured_gaze_vergence = np.array([])
target_gaze_vergence = np.array([])
stimulus_ids = np.array([])
stimulus_distance = np.array([])
subject_ids = np.array([])
for procedure, stimulus_position_z in zip(procedures, stimulus_positions_z):

    MAE_procedure = {}
    for index_subject, subject_ID in enumerate(subject_IDs):
        # load data
        loaded_data = np.load(f'data/{subject_ID}/analysis data/data_{procedure}.npz')
        left_visual_axis = loaded_data['left_visual_axis']
        right_visual_axis = loaded_data['right_visual_axis']
        pog_degrees = loaded_data['left_pog_degrees']
        left_cor = loaded_data['left_cor']
        right_cor = loaded_data['right_cor']
        time = loaded_data['time']
        is_outlier = loaded_data['is_outlier']

        # axs[counter, 0].plot(time, pog_degrees[:,0], label=f'{subject_ID}')
        # axs[counter, 1].plot(time, pog_degrees[:,1], label=f'{subject_ID}')
        # axs[counter, 0].set_ylim(-15, 15)
        # axs[counter, 1].set_ylim(-15, 15)

        left_ver = np.arcsin(left_visual_axis[:,1])
        left_hor = np.arcsin(left_visual_axis[:,0] / np.cos(left_ver))
        left_ver_deg = np.degrees(left_ver)
        left_hor_deg = np.degrees(left_hor)

        right_ver = np.arcsin(right_visual_axis[:,1])
        right_hor = np.arcsin(right_visual_axis[:,0] / np.cos(right_ver))
        right_ver_deg = np.degrees(right_ver)
        right_hor_deg = np.degrees(right_hor)


        stimulus_number = loaded_data['stimulus_number']
        stimulus_numbers = np.linspace(1, 5, 5)

        # position stimuli
        stimulus_positions = np.array([[70, -70, stimulus_position_z],
                                    [0, 0, stimulus_position_z],
                                    [-70, 70, stimulus_position_z],
                                    [-70, -70, stimulus_position_z],
                                    [70, 70, stimulus_position_z],
                                    ])

        # eye positions
        position_left_eye = np.array([30, 0, 650])
        position_right_eye = np.array([-30, 0, 650])

        # calculate target vergence for each stimulus
        target_vergences = [vergence_target(target, left_cor, right_cor) for target in stimulus_positions]
        target_vergences = np.array(target_vergences)

        gaze_vergence = np.degrees(np.arccos(np.sum(left_visual_axis * right_visual_axis, axis=1)))
        # gaze_vergence = np.abs(left_hor_deg - right_hor_deg)

        MAE_vergence = []
        for stimulus, target_vergence in zip(stimulus_numbers, target_vergences):
            stimulus_indices = np.where(stimulus_number == stimulus)
            if 1 in is_outlier[stimulus_indices]:
                print(subject_ID, procedure)
                continue
            else:    
                no_stimulus_datapoints = len(stimulus_indices[0])

                # calculate mean_vergence
                mean_gaze_vergence = np.nanmean(gaze_vergence[stimulus_indices], axis=0)
                MAE_vergence.append(np.abs(target_vergence - mean_gaze_vergence))

                if not np.isnan(mean_gaze_vergence):
                    measured_gaze_vergence = np.append(measured_gaze_vergence, mean_gaze_vergence)
                    target_gaze_vergence = np.append(target_gaze_vergence, target_vergence)
                    stimulus_ids = np.append(stimulus_ids, stimulus)
                    stimulus_distance = np.append(stimulus_distance, stimulus_position_z)
                    subject_ids = np.append(subject_ids, subject_ID)
                    


        MAE_procedure[subject_ID] = MAE_vergence

    counter += 1
    
    MAEs_vergence[procedure] = MAE_procedure

data = pd.DataFrame({
    'subject_id': subject_ids,
    'stimulus_id': stimulus_ids,
    'stimulus_distance': stimulus_distance,
    'measured_vergence': measured_gaze_vergence,
    'target_vergence': target_gaze_vergence,
})

# Ensure 'stimulus_distance' is treated as a categorical variable
data['stimulus_distance'] = data['stimulus_distance'].astype('category')

# for i in range(3):
#     axs[i, 0].set_ylabel('POG (degrees)')
# for i in range(2):
#     axs[2, i].set_xlabel('time (s)')

# plt.legend()
# plt.show()
# %%
# calculate mean std 
print("Grand average MAE")
ae_data = data['measured_vergence'] - data['target_vergence']
ae_data.abs().agg(['mean', 'std'])

# %%
# Verify that all columns have the same length
print(data.isnull().sum())


model = smf.mixedlm(
    "measured_vergence ~ target_vergence",
    data,
    groups=data['subject_id'], # random variable with random intercept
    # re_formula="1", # sets only random intercept (not needed according https://www.statsmodels.org/dev/generated/statsmodels.regression.mixed_linear_model.MixedLM.html#statsmodels.regression.mixed_linear_model.MixedLM)
)

result = model.fit()

print(result.summary())

# Define markers for each subject
subjects = data['subject_id'].unique()

# Plotting the data
plt.figure(figsize=(10, 6))

# scatter plt each subject with a unique marker and color
sns.scatterplot(x='target_vergence',
y='measured_vergence',
data=data,
hue='subject_id',
style='subject_id',
s=sns_marker_size,
)


# Plotting the regression line
# Get the fixed effect coefficients
intercept = -0.24576 # from matlab, from python: result.fe_params['Intercept']
slope = 0.98546 # from matlab, from python: result.fe_params['target_vergence']

# Generate x values for the regression line
x_vals = np.linspace(data['target_vergence'].min(), data['target_vergence'].max(), 100)

# from matlab:
random_effects = [
    -0.9395,
    -1.0802,
    0.8065,
    0.3938,
    0.8194,
]

for random_intercept in random_effects:
    y_vals = (intercept + random_intercept) + slope * x_vals
    plt.plot(x_vals, y_vals, zorder=0)

# Compute the predicted y values
y_vals = intercept + slope * x_vals

plt.plot(x_vals, y_vals, color='black', label='Regression', zorder=0)

# Plotting the identity line (y = x)
plt.plot(x_vals, x_vals, color='gray', linestyle='--', label='Identity', zorder=0)


# Customizing the plot
rcParams.update({'font.size': 16})
plt.xlabel(u'Target Vergence (\N{DEGREE SIGN})')
plt.ylabel(u'Measured Vergence (\N{DEGREE SIGN})')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.show()
# %%
