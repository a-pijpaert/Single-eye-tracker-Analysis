#%%
import os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, chi2
import statsmodels.formula.api as smf
import statsmodels.api as sm
from __plot_params import *

# %matplotlib qt

from helpers.recording_state import RecordingState

# set subject IDs
subject_IDs = ['s001',
               's002',
               's003',
               's007',
               's008',]

# set plotting params
plt.rcParams['font.size'] = font_size

puil_scaling_factor = 0.007795903499999998 # scaling factor used to convert pupil area in pixels to pupil diameter in mm

mean_pupil_sizes = []
mean_vergences = []
for subject_ID in subject_IDs:
    # set parameters
    procedure = RecordingState.VERGENCE_PUPIL

    # load data
    loaded_data = np.load(f'data/{subject_ID}/analysis data/data_{procedure.name.lower()}.npz')

    # get area of the pupil in pixels
    left_pupil_size = loaded_data['left_pupil_area']
    right_pupil_size = loaded_data['right_pupil_area']

    left_visual_axis = loaded_data['left_visual_axis']
    right_visual_axis = loaded_data['right_visual_axis']

    stimulus_number = loaded_data['stimulus_number']

    # calculate mean Average Pupil Areas and vergences
    mean_pupil_size = []
    mean_vergence = []
    for stimulus in np.linspace(1, 5, 5):
        stimulus_indices = np.where(stimulus_number == stimulus)

        # calculate mean Average Pupil Area over time
        mean_pupil_size.append(np.mean((left_pupil_size[stimulus_indices] + right_pupil_size[stimulus_indices]) / 2))

        # calculate mean_vergence
        mean_left_va = np.nanmean(left_visual_axis[stimulus_indices], axis=0)
        mean_right_va = np.nanmean(right_visual_axis[stimulus_indices], axis=0)
        mean_vergence.append(np.degrees(np.arccos(np.dot(mean_left_va, mean_right_va))))

    mean_pupil_sizes.append(mean_pupil_size)
    mean_vergences.append(mean_vergence)

    # plt.plot(mean_pupil_size, mean_vergence, label=f'{subject_ID}')

mean_pupil_diam_mm = 2*np.sqrt(puil_scaling_factor*np.array(mean_pupil_sizes)/np.pi)
# print(f'mean pupil sizes: {np.array([mean_pupil_sizes])}')
# plt.legend()
# plt.ylabel('Measured vergence (deg)')
# plt.xlabel('pupil area (pix)')
# Initialize an empty list to store rows of data
rows = []

# Iterate over each subject and their corresponding data
for i, subject in enumerate(subject_IDs):
    for j in range(5):  # Assuming there are 4 entries per subject
        rows.append([subject, mean_pupil_sizes[i][j], mean_pupil_diam_mm[i][j], mean_vergences[i][j]])

# Create a DataFrame with the collected rows and define column names
df = pd.DataFrame(rows, columns=['subject_id', 'average_pupil_area', 'average_pupil_diameter_mm', 'measured_vergence'])

# Drop rows with missing values
df.dropna(inplace=True)


# %% plot model
# Plot each subject with a unique marker and color
plt.figure(figsize=figure_size)
sns.scatterplot(x='average_pupil_area',
y='measured_vergence',
data=df,
hue='subject_id',
style='subject_id',
s=sns_marker_size
)

# Get the fixed effect coefficients
intercept = 5.0342
slope = -9.3807e-05

# Generate x values for the regression line
x_vals = np.linspace(df['average_pupil_area'].min(), df['average_pupil_area'].max(), 100)

# from matlab:
random_slopes = [
    -4.6296e-05,
    -0.0001,
    -0.0002,
    1.2834e-05,
    0.0004,
]

random_intercepts = [
    0.0409,
   -1.2509,
    1.7701,
    0.8633,
   -1.4234,
]

for random_intercept, random_slope in zip(random_intercepts, random_slopes):
    y_vals = (intercept + random_intercept) + (slope + random_slope) * x_vals
    plt.plot(x_vals, y_vals, zorder=0)

# Compute the predicted y values
y_vals = intercept + slope * x_vals

plt.plot(x_vals, y_vals, color='black', label='Regression', zorder=0)



# Customizing the plot
# rcParams.update({'font.size': 16})
plt.xlabel(u'Average Pupil Area (pixels)')
plt.ylabel(r'$ \alpha_{measured} (\degree)$')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.show()


# %% plot model using pupil diameter
# Plot each subject with a unique marker and color
plt.figure(figsize=figure_size)
sns.scatterplot(x='average_pupil_diameter_mm',
y='measured_vergence',
data=df,
hue='subject_id',
style='subject_id',
s=sns_marker_size
)

# Get the fixed effect coefficients
intercept = 5.0342
slope = -9.3807e-05

# Generate x values for the regression line
x_vals = np.linspace(df['average_pupil_diameter_mm'].min(), df['average_pupil_diameter_mm'].max(), 100)

# from matlab:
random_slopes = [
    -4.6296e-05,
    -0.0001,
    -0.0002,
    1.2834e-05,
    0.0004,
]

random_intercepts = [
    0.0409,
   -1.2509,
    1.7701,
    0.8633,
   -1.4234,
]

for random_intercept, random_slope in zip(random_intercepts, random_slopes):
    y_vals = (intercept + random_intercept) + (slope + random_slope) * x_vals
    plt.plot(x_vals, y_vals, zorder=0)

# Compute the predicted y values
y_vals = intercept + slope * x_vals

plt.plot(x_vals, y_vals, color='black', label='Regression', zorder=0)



# Customizing the plot
# rcParams.update({'font.size': 16})
plt.xlabel(u'Average Pupil Area (pixels)')
plt.ylabel(r'$ \alpha_{measured} (\degree)$')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.show()

