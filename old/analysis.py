import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

# Define the directory containing the subfolders
directory = '/home/arthur/Projects/AMESMC/data/eye_metrics'

# defin trial names
trial_names = ['mono_left', 'mono_right', 'bino_left', 'bino_right']

# Initialize a dictionary to store DataFrames for each subfolder
dataframes = {}

# Loop through each subfolder
for subdir in os.listdir(directory):
    subdir_path = os.path.join(directory, subdir)
    if os.path.isdir(subdir_path) and subdir in trial_names:
        # Initialize a list to store DataFrames for CSV files in the subfolder
        dfs = []
        # Loop through each CSV file in the subfolder
        for file in os.listdir(subdir_path):
            file_path = os.path.join(subdir_path, file)
            if file.endswith('.csv'):
                # Read the CSV file into a DataFrame
                df = pd.read_csv(file_path)
                df['subject_id'] = file[:4]
                # Append the DataFrame to the list
                dfs.append(df)
        # Store the list of DataFrames in the dictionary with the subfolder name as the key
        dataframes[subdir] = dfs


figure1 = plt.figure()
ax1 = figure1.add_subplot()
ax1.invert_xaxis()
ax1.invert_yaxis()
ax1.set_xlabel('Horizontal POG ($^\circ$)')
ax1.set_ylabel('Vertical POG ($^\circ$)')
ax1.set_title('Point of gaze estimates')

figure2, axs2 = plt.subplots(2,2)
figure2.suptitle('Horizontal POG estimates vs horizontal targets')
axs2 = axs2.flatten()
axs2[2].set_xlabel('Horizontal target location ($^\circ$)')
axs2[3].set_xlabel('Horizontal target location ($^\circ$)')
axs2[0].set_ylabel('Horizontal POG ($^\circ$)')
axs2[2].set_ylabel('Horizontal POG ($^\circ$)')

figure3, axs3 = plt.subplots(2,2)
figure3.suptitle('Vertical POG estimates vs vertical targets')
axs3 = axs3.flatten()
axs3[2].set_xlabel('Vertical target location ($^\circ$)')
axs3[3].set_xlabel('Vertical target location ($^\circ$)')
axs3[0].set_ylabel('Vertical POG ($^\circ$)')
axs3[2].set_ylabel('Vertical POG ($^\circ$)')

figure4, axs4 = plt.subplots(1,2)
figure4.suptitle('Accuracy box plots per horizontal target location')
axs4[0].set_xlabel('Horizontal target location ($^\circ$)')
axs4[0].set_ylabel('Horizontal MAE ($^\circ$)')
axs4[1].set_xlabel('Horizontal target location ($^\circ$)')
axs4[1].set_ylabel('Vertical MAE ($^\circ$)')

figure5, axs5 = plt.subplots(1,2)
figure5.suptitle('Precision (STD) box plots per horizontal target location')
axs5[0].set_xlabel('Horizontal target location ($^\circ$)')
axs5[0].set_ylabel('Horizontal STD precision ($^\circ$)')
axs5[1].set_xlabel('Horizontal target location ($^\circ$)')
axs5[1].set_ylabel('Vertical STD precision ($^\circ$)')

figure6, axs6 = plt.subplots(1,2)
figure6.suptitle('Precision (S2S) box plots per horizontal target location')
axs6[0].set_xlabel('Horizontal target location ($^\circ$)')
axs6[0].set_ylabel('Horizontal S2S precision ($^\circ$)')
axs6[1].set_xlabel('Horizontal target location ($^\circ$)')
axs6[1].set_ylabel('Vertical S2S precision ($^\circ$)')

figure7 = plt.figure()
ax7 = figure7.add_subplot()
ax7.invert_xaxis()
ax7.invert_yaxis()
ax7.set_xlabel('Horizontal target location ($^\circ$)')
ax7.set_ylabel('Vertical target location ($^\circ$)')
ax7.set_title('Vergence difference per target location')

def on_close(event):
    plt.close('all')

figure_names = ['pog', 'horizontal_pog', 'vertical_pog', 'accuracy', 'precision_std', 'precision_s2s', 'vergence']
figures = [figure1, figure2, figure3, figure4, figure5, figure6, figure7]
for figure in figures:
    figure.canvas.mpl_connect('close_event', on_close)

# List of predefined marker types
marker_types = ['o', 'X', 's', 'P', '>', '<', 'P', 'X', '*', '+']

# create a color palette and markers
palette_name = 'tab10'
num_datasets = len(dataframes['mono_left'])
colors = sns.color_palette(palette=palette_name,
                        n_colors=num_datasets)
markers = marker_types[:num_datasets]

# Initialize an empty list to store modified dataframes
modified_all_dfs = []
modified_mean_dfs = []

# Iterate through each trial
for trial_index, trial_name in enumerate(trial_names):
    trial_dataframes = dataframes[trial_name]
    if trial_name == 'bino_left':
        print('niks')
    
    # Iterate through each dataframe for the current trial
    for df in trial_dataframes:
        # Add a new column 'trial_name' with the current trial name
        df['trial_name'] = trial_name
        # Append the modified dataframe to the list
        modified_all_dfs.append(df)

    combined_df = pd.concat(trial_dataframes)
    headers_to_exclude = ['is_outlier', 'trial_name', 'subject_id'] 
    trial_headers = [header for header in combined_df.columns if header not in headers_to_exclude]
    filtered_df = combined_df[combined_df['is_outlier'] == False]
    outliers_df = combined_df[combined_df['is_outlier'] == True]
    # add a column for the vergence diff if trial_name == bino_left
    if trial_name == 'bino_left':
        vergence_headers = ['stimulus_id', 'target_x', 'target_y', 'vergence_target', 'vergence_gaze', 'vergence_diff_abs', 'subject_id']
        combined_df['vergence_diff_abs'] = (combined_df['vergence_target'] - combined_df['vergence_gaze']).abs()
        filtered_df = combined_df[combined_df['is_vergence_outlier'] == False]
        vergence_df = filtered_df.copy()[vergence_headers]
        mean_vergence_df = vergence_df.groupby('stimulus_id')[vergence_headers[:-1]].mean()
        mean_vergence_15deg_df = mean_vergence_df[(mean_vergence_df['target_x'] >= -15) & (mean_vergence_df['target_x'] <= 15)]


    mean_all_subjects_df = filtered_df.groupby('stimulus_id')[trial_headers].mean()
    mean_all_subjects_df['trial_name'] = trial_name
    modified_mean_dfs.append(mean_all_subjects_df)

# Concatenate all modified dataframes
combined_all_trial_df = pd.concat(modified_all_dfs)
combined_all_trial_df.reset_index(inplace=True)
filtered_combined_all_trial_df = combined_all_trial_df[combined_all_trial_df['is_outlier'] == False]
combined_means_df = pd.concat(modified_mean_dfs)

# Calculate absolute error for each row
filtered_combined_all_trial_df.loc[:, 'absolute_error_x'] = (filtered_combined_all_trial_df['target_x'] - filtered_combined_all_trial_df['mean_x']).abs()
filtered_combined_all_trial_df.loc[:, 'absolute_error_y'] = (filtered_combined_all_trial_df['target_y'] - filtered_combined_all_trial_df['mean_y']).abs()

# calculate accuracy for all stimuli
accuracy_all_df = filtered_combined_all_trial_df.groupby('trial_name')[['absolute_error_x', 'absolute_error_y']].agg(['mean', 'std'])
accuracy_all = filtered_combined_all_trial_df[['absolute_error_x', 'absolute_error_y']].agg(['mean', 'std'])

# filter out values that are not within -15 and 15 degrees on the horizontal axis
filtered_15deg_df = filtered_combined_all_trial_df[
    (filtered_combined_all_trial_df['target_x'] >= -15) &
    (filtered_combined_all_trial_df['target_x'] <= 15)
]

# calculate accuracy for stimuli within -15 and 15 degrees on the horizontal axis
accuracy_15deg_df = filtered_15deg_df.groupby('trial_name')[['absolute_error_x', 'absolute_error_y']].agg(['mean', 'std'])
accuracy_15deg = filtered_15deg_df[['absolute_error_x', 'absolute_error_y']].agg(['mean', 'std'])

# calculate STD precision for all stimuli
precision_std_df = filtered_combined_all_trial_df.groupby('trial_name')[['precision_x_std', 'precision_y_std']].agg(['mean', 'std'])
precision_std = filtered_combined_all_trial_df[['precision_x_std', 'precision_y_std']].agg(['mean', 'std'])

# calculate S2s precision for all stimuli
precision_s2s_df = filtered_combined_all_trial_df.groupby('trial_name')[['precision_x_s2s', 'precision_y_s2s']].agg(['mean', 'std'])
precision_s2s = filtered_combined_all_trial_df[['precision_x_s2s', 'precision_y_s2s']].agg(['mean', 'std'])

# vergence accuracy
vergence_all_accuracy = mean_vergence_df['vergence_diff_abs'].agg(['mean', 'std'])
vergence_15deg_accuracy = mean_vergence_15deg_df['vergence_diff_abs'].agg(['mean', 'std']) 

pd.options.display.float_format = '{:.2f}'.format
# print(f"Accuracy all \n{accuracy_all_df} \n")
# print(f"Accuracy 15 degrees \n{accuracy_15deg_df} \n")
# print(f"Precision STD \n{precision_std_df} \n")
# print(f"Precision S2S \n{precision_s2s_df} \n")
# print(f"{accuracy_all = }\n{accuracy_15deg = }\n{precision_std}\n{precision_s2s}")
print(f'vergence: \n{vergence_all_accuracy}\n{vergence_15deg_accuracy}')
data_frames = [accuracy_all_df,
               accuracy_15deg_df,
               precision_std_df,
               precision_s2s_df]


# with pd.ExcelWriter('data/data.xlsx') as writer:   
#     for index, df in enumerate(data_frames):
#         df.columns = ['_'.join(col).strip() for col in df.columns.values]
#         df.to_excel(writer, sheet_name=f'Sheet{index}', index=False)


# accuracy plot of all tests
sns.scatterplot(data=mean_all_subjects_df, 
                x='target_x', 
                y='target_y', 
                ax=ax1, 
                color='black')
sns.scatterplot(data=combined_means_df,
                x='mean_x',
                y='mean_y',
                ax=ax1,
                hue='trial_name',
                style='trial_name')

# horizontal plot per test
for trial_index, trial_name in enumerate(trial_names):
    trial_means_df = combined_means_df[combined_means_df['trial_name'] == trial_name]
    axs2[trial_index].axline((0,0), slope=1, color='gray') # identity line
    sns.scatterplot(data=combined_means_df, 
                    x='target_x', 
                    y='mean_x', 
                    ax=axs2[trial_index],
                    color=colors[trial_index],
                    marker=markers[trial_index],
                    label=f'{trial_name}')
axs2[0].set_xlabel(None)
axs2[1].set_xlabel(None)
axs2[1].set_ylabel(None)
axs2[3].set_ylabel(None)

# horizontal plot per test
for trial_index, trial_name in enumerate(trial_names):
    trial_means_df = combined_means_df[combined_means_df['trial_name'] == trial_name]
    axs3[trial_index].axline((0,0), slope=1, color='gray') # identity line
    sns.scatterplot(data=combined_means_df, 
                    x='target_y', 
                    y='mean_y', 
                    ax=axs3[trial_index],
                    color=colors[trial_index],
                    marker=markers[trial_index],
                    label=f'{trial_name}')
axs3[0].set_xlabel(None)
axs3[1].set_xlabel(None)
axs3[1].set_ylabel(None)
axs3[3].set_ylabel(None)
    
# accuracy boxplots
sns.boxplot(data=filtered_combined_all_trial_df,
            x='target_x',
            y='accuracy_x',
            hue='trial_name',
            ax=axs4[0],
            fliersize=2)
sns.boxplot(data=filtered_combined_all_trial_df,
            x='target_x',
            y='accuracy_y',
            hue='trial_name',
            ax=axs4[1],
            fliersize=2)
figure4.tight_layout()  # Adjust the width space between subplots

# Precision STD boxplots
sns.boxplot(data=filtered_combined_all_trial_df,
            x='target_x',
            y='precision_x_std',
            hue='trial_name',
            ax=axs5[0],
            fliersize=2)
sns.boxplot(data=filtered_combined_all_trial_df,
            x='target_x',
            y='precision_y_std',
            hue='trial_name',
            ax=axs5[1],
            fliersize=2)
figure5.tight_layout()  # Adjust the width space between subplots

# Precision S2S boxplots
sns.boxplot(data=filtered_combined_all_trial_df,
            x='target_x',
            y='precision_x_s2s',
            hue='trial_name',
            ax=axs6[0],
            fliersize=2)
sns.boxplot(data=filtered_combined_all_trial_df,
            x='target_x',
            y='precision_y_s2s',
            hue='trial_name',
            ax=axs6[1],
            fliersize=2)
figure6.tight_layout()  # Adjust the width space between subplots


# vergence plot
ax7.errorbar(x=mean_vergence_df['target_x'],
             y=mean_vergence_df['target_y'],
             yerr=mean_vergence_df['vergence_diff_abs'],
             fmt='none', 
             ecolor=colors[0], 
             capsize=5)
sns.scatterplot(data=mean_vergence_df, 
                x='target_x', 
                y='target_y', 
                ax=ax7, 
                color='black')

# plt.tight_layout()
plt.show()

for figure, figure_name in zip(figures, figure_names):
    figure.savefig(f'{figure_name}.png')
    print(f'figure {figure_name}.png saved!')
