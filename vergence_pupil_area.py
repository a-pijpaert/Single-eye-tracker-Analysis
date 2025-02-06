#%%
import os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, chi2
import scipy
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
mean_all_left_pog = []
mean_all_right_pog = []
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

    left_pog_deg = loaded_data['left_pog_degrees']
    right_pog_deg = loaded_data['right_pog_degrees']

    # calculate mean Average Pupil Areas and vergences
    mean_pupil_size = []
    mean_vergence = []
    mean_left_pog_deg = []
    mean_right_pog_deg = []
    for stimulus in np.linspace(1, 5, 5):
        stimulus_indices = np.where(stimulus_number == stimulus)

        # calculate mean Average Pupil Area over time
        mean_pupil_size.append(np.mean((left_pupil_size[stimulus_indices] + right_pupil_size[stimulus_indices]) / 2))

        # calculate mean_vergence
        mean_left_va = np.nanmean(left_visual_axis[stimulus_indices], axis=0)
        mean_right_va = np.nanmean(right_visual_axis[stimulus_indices], axis=0)
        mean_vergence.append(np.degrees(np.arccos(np.dot(mean_left_va, mean_right_va))))

        #calculate mean pog in degrees
        mean_left_pog_deg.append(np.nanmean(left_pog_deg[stimulus_indices], axis=0))
        mean_right_pog_deg.append(np.nanmean(right_pog_deg[stimulus_indices], axis=0))

    mean_pupil_sizes.append(mean_pupil_size)
    mean_vergences.append(mean_vergence)

    mean_all_left_pog.append(mean_left_pog_deg)
    mean_all_right_pog.append(mean_right_pog_deg)

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
        rows.append([subject,
        mean_pupil_sizes[i][j],
        mean_pupil_diam_mm[i][j],
        mean_vergences[i][j],
        mean_all_left_pog[i][j][0],
        mean_all_left_pog[i][j][1],
        mean_all_right_pog[i][j][0],
        mean_all_right_pog[i][j][1]])

# Create a DataFrame with the collected rows and define column names
df = pd.DataFrame(rows, columns=['subject_id',
    'average_pupil_area',
    'average_pupil_diameter_mm',
    'measured_vergence',
    'left_pog_x_deg',
    'left_pog_y_deg',
    'right_pog_x_deg',
    'right_pog_y_deg'])

# Drop rows with missing values
df.dropna(inplace=True)


# %% Compute Spearman correlation for each subject
results = []
for subject, group in df.groupby('subject_id'):
    rho, p_value = spearmanr(group['measured_vergence'], group['average_pupil_area'])
    results.append({'Subject': subject, 'Spearman_Rho': rho, 'P_Value': p_value})

# Convert results to DataFrame
results_df = pd.DataFrame(results)

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
intercept = 5.29538964143865
slope = -0.105442222443439

# Generate x values for the regression line
x_vals = np.linspace(df['average_pupil_diameter_mm'].min(), df['average_pupil_diameter_mm'].max(), 100)

# from matlab:
random_slopes = [
    -0.0981213556631409,
    -0.178134473394235,
    -0.295272370314246,
    0.0141548023371434,
    0.557373397034479,
]

random_intercepts = [
    0.504627586639085,
    -0.538458129276204,
    2.80677927627584,
    0.841303086409537,
    -3.61425182004856,
]

for random_intercept, random_slope in zip(random_intercepts, random_slopes):
    y_vals = (intercept + random_intercept) + (slope + random_slope) * x_vals
    plt.plot(x_vals, y_vals, zorder=0)

# Compute the predicted y values
y_vals = intercept + slope * x_vals

plt.plot(x_vals, y_vals, color='black', label='Regression', zorder=0)



# Customizing the plot
# rcParams.update({'font.size': 16})
plt.xlabel(u'Average Pupil Diameter (mm)')
plt.ylabel(r'$ \alpha_{measured} (\degree)$')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.show()

# %% plot model
# Plot each subject with a unique marker and color
figure, axs = plt.subplots(2, 2, figsize=figure_size)
axs = axs.flatten()
gaze_components = ['left_pog_x_deg', 'left_pog_y_deg', 'right_pog_x_deg', 'right_pog_y_deg']
fixed_intercepts = [-0.17279, -2.2958, 0.76366, -1.4125]
fixed_slopes = [7.2615e-05, 0.00051295, -0.00019947, 9.4734e-05]
random_intercepts = [[-0.36313, 0.81475, -0.81381, -0.28824, 0.65042],
            [0.3421, -0.58601, 0.43272, -0.1084, -0.080402],
            [0.50474, 0.18892, -0.098503, 0.0046856, -0.59985],
            [0.20467, 0.34334, 0.7907, -0.42743, -0.91128]]
random_slopes = [[-3.9534e-06, 8.8704e-06, -8.8601e-06, -3.1381e-06, 7.0813e-06],
            [3.7342e-05, -6.3966e-05, 4.7234e-05, -1.1833e-05, -8.7764e-06],
            [-0.00025015, -9.3631e-05, 4.8818e-05, -2.3222e-06, 0.00029729],
            [2.1928e-06, 3.6785e-06, 8.4716e-06, -4.5795e-06, -9.7635e-06]]

for index, gaze_comp in enumerate(gaze_components):
    sns.scatterplot(x='average_pupil_area',
    y=gaze_comp,
    data=df,
    hue='subject_id',
    style='subject_id',
    s=sns_marker_size,
    ax=axs[index]
    )

    # Get the fixed effect coefficients
    intercept = fixed_intercepts[index]
    slope = fixed_slopes[index]

    # Generate x values for the regression line
    x_vals = np.linspace(df['average_pupil_area'].min(), df['average_pupil_area'].max(), 100)

    # from matlab:
    slopes = random_slopes[index]
    intercepts = random_intercepts[index]

    for random_intercept, random_slope in zip(intercepts, slopes):
        y_vals = (intercept + random_intercept) + (slope + random_slope) * x_vals
        axs[index].plot(x_vals, y_vals, zorder=0)

    # Compute the predicted y values
    y_vals = intercept + slope * x_vals

    axs[index].plot(x_vals, y_vals, color='black', label='Regression', zorder=0)



    # Customizing the plot
    # rcParams.update({'font.size': 16})
    axs[index].set_xlabel(u'Average Pupil Area (pixels)')
    axs[index].set_ylabel(f'{gaze_comp} $(\degree)$')
    axs[index].legend(loc='upper left', bbox_to_anchor=(1, 1))
    axs[index].get_legend().remove()
plt.tight_layout()
plt.show()

# %% Analysis pupil size on gaze
X = df[['left_pog_x_deg', 'left_pog_y_deg', 'right_pog_x_deg', 'right_pog_y_deg']]
y = df['average_pupil_area']

# Add a constant (intercept) to the independent variables
X_sm = sm.add_constant(X)  # Adds a column of ones for the intercept

# Fit Ordinary Least Squares (OLS) regression
model = sm.OLS(y, X_sm).fit()

# Print the regression summary (R², p-values, etc.)
print(model.summary())

# Create subplots (2 rows, 2 columns)
fig, axs = plt.subplots(2, 2, figsize=(14, 12))

# Scatter plot for left_pog_x_deg vs average_pupil_area
sns.regplot(y='left_pog_x_deg', x='average_pupil_area', data=df, scatter_kws={'s': 100, 'alpha': 0.5}, line_kws={'color': 'red'}, ax=axs[0, 0])
axs[0, 0].set_title('Left Gaze X-Position vs Average Pupil Area')
axs[0, 0].set_ylabel('Left POG X (degrees)')
axs[0, 0].set_xlabel('Average Pupil Area (pixels)')

# Scatter plot for left_pog_y_deg vs average_pupil_area
sns.regplot(y='left_pog_y_deg', x='average_pupil_area', data=df, scatter_kws={'s': 100, 'alpha': 0.5}, line_kws={'color': 'red'}, ax=axs[0, 1])
axs[0, 1].set_title('Left Gaze Y-Position vs Average Pupil Area')
axs[0, 1].set_ylabel('Left POG Y (degrees)')
axs[0, 1].set_xlabel('Average Pupil Area (pixels)')

# Scatter plot for right_pog_x_deg vs average_pupil_area
sns.regplot(y='right_pog_x_deg', x='average_pupil_area', data=df, scatter_kws={'s': 100, 'alpha': 0.5}, line_kws={'color': 'red'}, ax=axs[1, 0])
axs[1, 0].set_title('Right Gaze X-Position vs Average Pupil Area')
axs[1, 0].set_ylabel('Right POG X (degrees)')
axs[1, 0].set_xlabel('Average Pupil Area (pixels)')

# Scatter plot for right_pog_y_deg vs average_pupil_area
sns.regplot(y='right_pog_y_deg', x='average_pupil_area', data=df, scatter_kws={'s': 100, 'alpha': 0.5}, line_kws={'color': 'red'}, ax=axs[1, 1])
axs[1, 1].set_title('Right Gaze Y-Position vs Average Pupil Area')
axs[1, 1].set_ylabel('Right POG Y (degrees)')
axs[1, 1].set_xlabel('Average Pupil Area (pixels)')

# Adjust layout
plt.tight_layout()
plt.show()


# Create subplots (2 rows, 2 columns)
fig, axs = plt.subplots(2, 2, figsize=(14, 12))

# Residuals plot for left_pog_x_deg
sns.residplot(x='left_pog_x_deg', y='average_pupil_area', data=df, lowess=True, line_kws={'color': 'red'}, ax=axs[0, 0])
axs[0, 0].set_title('Residuals of Left Gaze X-Position vs Average Pupil Area')
axs[0, 0].set_xlabel('Left POG X (degrees)')
axs[0, 0].set_ylabel('Residuals')

# Residuals plot for left_pog_y_deg
sns.residplot(x='left_pog_y_deg', y='average_pupil_area', data=df, lowess=True, line_kws={'color': 'red'}, ax=axs[0, 1])
axs[0, 1].set_title('Residuals of Left Gaze Y-Position vs Average Pupil Area')
axs[0, 1].set_xlabel('Left POG Y (degrees)')
axs[0, 1].set_ylabel('Residuals')

# Residuals plot for right_pog_x_deg
sns.residplot(x='right_pog_x_deg', y='average_pupil_area', data=df, lowess=True, line_kws={'color': 'red'}, ax=axs[1, 0])
axs[1, 0].set_title('Residuals of Right Gaze X-Position vs Average Pupil Area')
axs[1, 0].set_xlabel('Right POG X (degrees)')
axs[1, 0].set_ylabel('Residuals')

# Residuals plot for right_pog_y_deg
sns.residplot(x='right_pog_y_deg', y='average_pupil_area', data=df, lowess=True, line_kws={'color': 'red'}, ax=axs[1, 1])
axs[1, 1].set_title('Residuals of Right Gaze Y-Position vs Average Pupil Area')
axs[1, 1].set_xlabel('Right POG Y (degrees)')
axs[1, 1].set_ylabel('Residuals')

# Adjust layout
plt.tight_layout()
plt.show()


# %% Save data
# import scipy
# Convert DataFrame to a dictionary
data_dict = {col: df[col].values for col in df.columns}

# Save to a .mat file
scipy.io.savemat('data_vergence_pupil.mat', data_dict)
# %%
