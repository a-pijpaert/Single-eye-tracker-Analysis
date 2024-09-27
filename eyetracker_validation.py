#%%
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
# import statsmodels.formula.api as smf
import seaborn as sns
# %matplotlib qt

from helpers.recording_state import RecordingState
from helpers.vergence_target import vergence_target
from helpers.calculate_rms_s2s import calculate_rms_s2s
from helpers.print_mae_info import print_mae_info

# set subject IDs
subject_IDs = ['s001',
               's003',
               's004',
               's005',
               's006',
               's007',]

# set parameters
procedures = [RecordingState.MEASUREMENT.name,
              RecordingState.MEASUREMENT_LEFT.name,
              RecordingState.MEASUREMENT_RIGHT.name,]

procedure = procedures[0]
subject_ID = subject_IDs[0]
# iterate over all stimuli
mean_pogx_subject = np.array([])
mean_pogy_subject = np.array([])
stim_posx_subject = np.array([])
stim_posy_subject = np.array([])
stimulus_ids = np.array([])
subject_ids = np.array([])
procedure_ids = np.array([])
outliers = np.array([])
eye_id = np.array([])
precision_x_SDs = np.array([])
precision_y_SDs = np.array([])
precision_x_S2Ss = np.array([])
precision_y_S2Ss = np.array([])
ae_x = np.array([])
ae_y = np.array([])

for procedure in procedures:
    stimulus_numbers = np.linspace(1,35,35)

    if procedure == RecordingState.MEASUREMENT.name:
        eyes = ['left', 'right']
        ocular = 'bino'
        print('kom ik hier')
    if procedure == RecordingState.MEASUREMENT_LEFT.name:
        eyes = ['left']
        ocular = 'mono'
    if procedure == RecordingState.MEASUREMENT_RIGHT.name:
        eyes = ['right']
        ocular = 'mono'

    for eye in eyes:
        print(f'{eye}_pog_degrees | {procedure} | {eyes}')
        for subject_ID in subject_IDs:
            # load data
            loaded_data = np.load(f'data/{subject_ID}/analysis data/data_{procedure.lower()}.npz')
            pog_degrees = loaded_data[f'{eye}_pog_degrees']
            time = loaded_data['time']
            is_outlier = loaded_data['is_outlier']
            stim_degrees = loaded_data['stim_pos_degrees']
            stimulus_number = loaded_data['stimulus_number']

            for stimulus in stimulus_numbers:
                stimulus_indices = np.where(stimulus_number == stimulus)[0]

                mean_pog_deg = np.nanmean(pog_degrees[stimulus_indices,:2], axis=0)
                precision_SD = np.nanstd(pog_degrees[stimulus_indices,:2], axis=0)
                precision_S2S = calculate_rms_s2s(pog_degrees[stimulus_indices,:2])
                stim_pos_deg = stim_degrees[stimulus_indices[0]]
                outlier = is_outlier[stimulus_indices[0]]
                ae = np.abs(mean_pog_deg-stim_pos_deg)
                vae = np.linalg.norm(mean_pog_deg-stim_pos_deg)

                if not np.isnan(mean_pog_deg[0]) or not np.isnan(mean_pog_deg[1]):
                    mean_pogx_subject = np.append(mean_pogx_subject, mean_pog_deg[0])
                    mean_pogy_subject = np.append(mean_pogy_subject, mean_pog_deg[1])
                    ae_x = np.append(ae_x, ae[0])
                    ae_y = np.append(ae_y, ae[1])
                    precision_x_SDs = np.append(precision_x_SDs, precision_SD[0])
                    precision_y_SDs = np.append(precision_y_SDs, precision_SD[1])
                    precision_x_S2Ss = np.append(precision_x_S2Ss, precision_S2S[0])
                    precision_y_S2Ss = np.append(precision_y_S2Ss, precision_S2S[1])
                    stim_posx_subject = np.append(stim_posx_subject, stim_pos_deg[0])
                    stim_posy_subject = np.append(stim_posy_subject, stim_pos_deg[1])
                    stimulus_ids = np.append(stimulus_ids, stimulus)
                    subject_ids = np.append(subject_ids, subject_ID)
                    procedure_ids = np.append(procedure_ids, procedure)
                    outliers = np.append(outliers, outlier)
                    eye_id = np.append(eye_id, f'{ocular} {eye}')

data = pd.DataFrame({
    'subject ID': subject_ids,
    'stimulus ID': stimulus_ids,
    'procedure': procedure_ids,
    'pog x': mean_pogx_subject,
    'pog y': mean_pogy_subject,
    'ae x': ae_x,
    'ae y': ae_y,
    'precision x SD': precision_x_SDs,
    'precision y SD': precision_y_SDs,
    'precision x S2S': precision_x_S2Ss,
    'precision y S2S': precision_y_S2Ss,
    'stimulus position x': stim_posx_subject,
    'stimulus position y': stim_posy_subject,
    'outlier': outliers,
    'eye': eye_id,

})

data_single_subject = data[data['subject ID'] == 's003']
data_stimuli = data_single_subject[data_single_subject['procedure'] == 'MEASUREMENT_RIGHT']
data_no_outliers = data[data['outlier'] == 0]
data_no_outliers_central = data_no_outliers[data_no_outliers['stimulus position x'].between(-15, 15)]

mae_subject = data_no_outliers.groupby(['subject ID', 'eye'])[['ae x', 'ae y']].mean().reset_index()
mae_subject_central = data_no_outliers_central.groupby(['subject ID', 'eye'])[['ae x', 'ae y']].mean().reset_index()

sd_subject = data_no_outliers.groupby(['subject ID', 'eye'])[['precision x SD', 'precision y SD']].mean().reset_index()
sd_subject_central = data_no_outliers_central.groupby(['subject ID', 'eye'])[['precision x SD', 'precision y SD']].mean().reset_index()

s2s_subject = data_no_outliers.groupby(['subject ID', 'eye'])[['precision x S2S', 'precision y S2S']].mean().reset_index()
s2s_subject_central = data_no_outliers_central.groupby(['subject ID', 'eye'])[['precision x S2S', 'precision y S2S']].mean().reset_index()


#%%
# Scatter plot of a single subject
figure1 = plt.figure(figsize=(10, 6))
sns.scatterplot(x='stimulus position x',
                y='stimulus position y',
                data=data_stimuli,
                color='black',
                s=50,
                figure=figure1)
sns.scatterplot(x='pog x',
                y='pog y',
                data=data_single_subject,
                hue='eye',
                style='eye',
                figure=figure1)

axis1 = figure1.axes[0]
axis1.set_xlim([-30,30])
axis1.set_ylim([-20,20])

axis1.set_ylabel('Vertical POG ($^\circ$)')
axis1.set_xlabel('Horizontal POG ($^\circ$)')

handles, labels = axis1.get_legend_handles_labels()
new_labels = [label.replace('subject ID', '').strip() for label in labels]
axis1.legend(handles, new_labels, title='', loc='upper left', bbox_to_anchor=(1, 1))


#%%
# horizontal pog vs target 
figure2, axs2 = plt.subplots(1,2, figsize=(10,6))

# draw identity lines
axs2[0].axline((0,0), slope=1, color='gray', zorder=0)
axs2[1].axline((0,0), slope=1, color='gray', zorder=0)
# axs2[1,0].axline((0,0), slope=1, color='gray', zorder=0)
# axs2[1,1].axline((0,0), slope=1, color='gray', zorder=0)

sns.scatterplot(data=data_no_outliers[data_no_outliers['eye'].isin(['mono left', 'mono right'])], 
                x='stimulus position x', y='pog x',
                hue='subject ID', style='subject ID', 
                s=100, ax=axs2[0])
sns.scatterplot(data=data_no_outliers[data_no_outliers['eye'].isin(['bino left', 'bino right'])], 
                x='stimulus position x', y='pog x',
                hue='subject ID', style='subject ID', 
                s=100, ax=axs2[1])
# sns.scatterplot(data=data_no_outliers[data_no_outliers['eye'] == 'mono left'], 
#                 x='stimulus position x', y='pog x',
#                 hue='subject ID', style='subject ID', 
#                 s=100, ax=axs2[1,0])
# sns.scatterplot(data=data_no_outliers[data_no_outliers['eye'] == 'mono right'], 
#                 x='stimulus position x', y='pog x',
#                 hue='subject ID', style='subject ID', 
#                 s=100, ax=axs2[1,1])


axs2[0].legend_.remove()
axs2[0].set_xlabel('Horizontal Stimulus Position ($^\circ$)')
axs2[0].set_ylabel('Horizontal POG ($^\circ$)')

axs2[1].legend_.remove()
axs2[1].set_xlabel('Horizontal Stimulus Position ($^\circ$)')
axs2[1].set_ylabel('')

# axs2[1,0].legend_.remove()

# axs2[1,1].set_ylabel('')
# axs2[1,1].legend_.remove()

# Defining custom 'xlim' and 'ylim' values.
custom_xlim = (-24, 24)
custom_ylim = (-35, 35)

# Setting the values for all axes.
plt.setp(axs2, xlim=custom_xlim, ylim=custom_ylim)

handles, labels = axs2[1].get_legend_handles_labels()
new_labels = [label.replace('subject ID', '').strip() for label in labels]
axs2[1].legend(handles, new_labels, title='', loc='upper left', bbox_to_anchor=(1, 1))

#%%
# vertical pog vs target 
figure3, axs3 = plt.subplots(1,2, figsize=(10,6))

# draw identity lines
axs3[0].axline((0,0), slope=1, color='gray', zorder=0)
axs3[1].axline((0,0), slope=1, color='gray', zorder=0)
# axs3[1,0].axline((0,0), slope=1, color='gray', zorder=0)
# axs3[1,1].axline((0,0), slope=1, color='gray', zorder=0)

sns.scatterplot(data=data_no_outliers[data_no_outliers['eye'].isin(['mono left', 'mono right'])], 
                x='stimulus position y', y='pog y',
                hue='subject ID', style='subject ID', 
                s=100, ax=axs3[0])
sns.scatterplot(data=data_no_outliers[data_no_outliers['eye'].isin(['bino left', 'bino right'])], 
                x='stimulus position y', y='pog y',
                hue='subject ID', style='subject ID', 
                s=100, ax=axs3[1])
# sns.scatterplot(data=data_no_outliers[data_no_outliers['eye'] == 'mono left'], 
#                 x='stimulus position y', y='pog y',
#                 hue='subject ID', style='subject ID', 
#                 s=100, ax=axs3[1,0])
# sns.scatterplot(data=data_no_outliers[data_no_outliers['eye'] == 'mono right'], 
#                 x='stimulus position y', y='pog y',
#                 hue='subject ID', style='subject ID', 
#                 s=100, ax=axs3[1,1])



axs3[0].legend_.remove()
axs3[0].set_xlabel('Vertical Stimulus Position ($^\circ$)')
axs3[0].set_ylabel('Vertical POG ($^\circ$)')

axs3[1].legend_.remove()
axs3[1].set_xlabel('Vertical Stimulus Position ($^\circ$)')
axs3[1].set_ylabel('')

# axs3[1,0].legend_.remove()

# axs3[1,1].set_ylabel('')
# axs3[1,1].legend_.remove()

# Defining custom 'xlim' and 'ylim' values.
custom_xlim = (-12, 12)
custom_ylim = (-25, 25)

# Setting the values for all axes.
plt.setp(axs3, xlim=custom_xlim, ylim=custom_ylim)

handles, labels = axs3[1].get_legend_handles_labels()
new_labels = [label.replace('subject ID', '').strip() for label in labels]
axs3[1].legend(handles, new_labels, title='', loc='upper left', bbox_to_anchor=(1, 1))

# %%
# MAE per procedure split between x and y
# Melt the DataFrame to long format
melted_mae = pd.melt(mae_subject, id_vars=['eye'], value_vars=['ae x', 'ae y'], 
                      var_name='mae', value_name='value')
melted_mae['mae'] = melted_mae['mae'].replace({'ae x': 'x', 'ae y': 'y'})

melted_mae_central = pd.melt(mae_subject_central, id_vars=['eye'], value_vars=['ae x', 'ae y'], 
                      var_name='mae', value_name='value')
melted_mae_central['mae'] = melted_mae_central['mae'].replace({'ae x': 'x', 'ae y': 'y'})

figure2, axs2 = plt.subplots(1,2, figsize=(12, 6))
sns.boxplot(x='mae', y='value', hue='eye', data=melted_mae,
            ax=axs2[0], showfliers=False)
sns.stripplot(x='mae', y='value', hue='eye', data=melted_mae, 
              dodge=True, alpha=0.7, ax=axs2[0], size=10, 
              edgecolor='white', linewidth=2, legend=False)
axs2[0].set_ylim([0, 3])
axs2[0].set_xlabel('')  
axs2[0].set_ylabel('')
axs2[0].legend_.remove()

sns.boxplot(x='mae', y='value', hue='eye', data=melted_mae_central,
            ax=axs2[1], showfliers=False)
sns.stripplot(x='mae', y='value', hue='eye', data=melted_mae_central, 
              dodge=True, alpha=0.7, ax=axs2[1], size=10, 
              edgecolor='white', linewidth=2, legend=False)
axs2[1].set_ylim([0, 3])
axs2[1].set_xlabel('')
axs2[1].set_ylabel('')

# Simplify legend by removing the word "eye"
handles, labels = axs2[1].get_legend_handles_labels()
new_labels = [label.replace('eye', '').strip() for label in labels]
axs2[1].legend(handles, new_labels, title='', loc='upper left', bbox_to_anchor=(1, 1))

figure2.align_ylabels()

# Set a shared y-label
figure2.text(0.04, 0.5, 'MAE ($^\circ$)', va='center', rotation='vertical', fontsize=12)

plt.show()

mae_summary = melted_mae.groupby(['eye', 'mae'])['value'].agg(['mean', 'std']).reset_index().sort_values('mae')
mae_summary_cenrtal = melted_mae_central.groupby(['eye', 'mae'])['value'].agg(['mean', 'std']).reset_index().sort_values('mae')

print('All points:')
print_mae_info(mae_summary)
print('')
print('Central points:')
print_mae_info(mae_summary_cenrtal)

#%%
# MAE per monocular and binocular with both eye together
melted_mae_bino_mono = melted_mae.copy()
melted_mae_bino_mono['eye'] = melted_mae_bino_mono['eye'].replace({'bino left': 'binocular', 'bino right': 'binocular', 'mono left': 'monocular', 'mono right': 'monocular'})

melted_mae_bino_mono_central = melted_mae_central.copy()
melted_mae_bino_mono_central['eye'] = melted_mae_bino_mono_central['eye'].replace({'bino left': 'binocular', 'bino right': 'binocular', 'mono left': 'monocular', 'mono right': 'monocular'})

figure3, axs3 = plt.subplots(1,2, figsize=(12, 6))
sns.boxplot(x='mae', y='value', hue='eye', data=melted_mae_bino_mono,
            ax=axs3[0], showfliers=False)
sns.stripplot(x='mae', y='value', hue='eye', data=melted_mae_bino_mono, 
              dodge=True, alpha=0.7, ax=axs3[0], size=10, 
              edgecolor='white', linewidth=2, legend=False)


axs3[0].set_ylim([0, 3])
axs3[0].set_xlabel('')  
axs3[0].set_ylabel('')
axs3[0].legend_.remove()

sns.boxplot(x='mae', y='value', hue='eye', data=melted_mae_bino_mono_central,
            ax=axs3[1], showfliers=False)
sns.stripplot(x='mae', y='value', hue='eye', data=melted_mae_bino_mono_central, 
              dodge=True, alpha=0.7, ax=axs3[1], size=10, 
              edgecolor='white', linewidth=2, legend=False)
axs3[1].set_ylim([0, 3])
axs3[1].set_xlabel('')
axs3[1].set_ylabel('')

# Simplify legend by removing the word "eye"
handles, labels = axs3[1].get_legend_handles_labels()
new_labels = [label.replace('eye', '').strip() for label in labels]
axs3[1].legend(handles, new_labels, title='', loc='upper left', bbox_to_anchor=(1, 1))

figure3.align_ylabels()

# Set a shared y-label
figure3.text(0.04, 0.5, 'MAE ($^\circ$)', va='center', rotation='vertical', fontsize=12)

plt.show()

mae_summary_bino_mono = melted_mae_bino_mono.groupby(['eye', 'mae'])['value'].agg(['mean', 'std']).reset_index().sort_values('eye')
mae_summary_bino_mono_cenrtal = melted_mae_bino_mono_central.groupby(['eye', 'mae'])['value'].agg(['mean', 'std']).reset_index().sort_values('eye')

print('All points:')
print_mae_info(mae_summary_bino_mono)
print('')
print('Central points:')
print_mae_info(mae_summary_bino_mono_cenrtal)


#%%
# SD per monocular and binocular with both eye together
melted_sd = pd.melt(sd_subject, id_vars=['eye'], value_vars=['precision x SD', 'precision y SD'], 
                      var_name='sd', value_name='value')
melted_sd['sd'] = melted_sd['sd'].replace({'precision x SD': 'x', 'precision y SD': 'y'})

melted_sd_central = pd.melt(sd_subject_central, id_vars=['eye'], value_vars=['precision x SD', 'precision y SD'], 
                      var_name='sd', value_name='value')
melted_sd_central['sd'] = melted_sd_central['sd'].replace({'precision x SD': 'x', 'precision y SD': 'y'})

melted_sd_bino_mono = melted_sd.copy()
melted_sd_bino_mono['eye'] = melted_sd_bino_mono['eye'].replace({'bino left': 'binocular', 'bino right': 'binocular', 'mono left': 'monocular', 'mono right': 'monocular'})

melted_sd_bino_mono_central = melted_sd_central.copy()
melted_sd_bino_mono_central['eye'] = melted_sd_bino_mono_central['eye'].replace({'bino left': 'binocular', 'bino right': 'binocular', 'mono left': 'monocular', 'mono right': 'monocular'})

figure3, axs3 = plt.subplots(1,2, figsize=(12, 6))
sns.boxplot(x='sd', y='value', hue='eye', data=melted_sd_bino_mono,
            ax=axs3[0], showfliers=False)
sns.stripplot(x='sd', y='value', hue='eye', data=melted_sd_bino_mono, 
              dodge=True, alpha=0.7, ax=axs3[0], size=10, 
              edgecolor='white', linewidth=2, legend=False)


axs3[0].set_ylim([0, .8])
axs3[0].set_xlabel('')  
axs3[0].set_ylabel('')
axs3[0].legend_.remove()

sns.boxplot(x='sd', y='value', hue='eye', data=melted_sd_bino_mono_central,
            ax=axs3[1], showfliers=False)
sns.stripplot(x='sd', y='value', hue='eye', data=melted_sd_bino_mono_central, 
              dodge=True, alpha=0.7, ax=axs3[1], size=10, 
              edgecolor='white', linewidth=2, legend=False)
axs3[1].set_ylim([0, .8])
axs3[1].set_xlabel('')
axs3[1].set_ylabel('')

# Simplify legend by removing the word "eye"
handles, labels = axs3[1].get_legend_handles_labels()
new_labels = [label.replace('eye', '').strip() for label in labels]
axs3[1].legend(handles, new_labels, title='', loc='upper left', bbox_to_anchor=(1, 1))

figure3.align_ylabels()

# Set a shared y-label
figure3.text(0.04, 0.5, 'sd ($^\circ$)', va='center', rotation='vertical', fontsize=12)

plt.show()

sd_summary_bino_mono = melted_sd_bino_mono.groupby(['eye', 'sd'])['value'].agg(['mean', 'std']).reset_index().sort_values('sd')
sd_summary_bino_mono_cenrtal = melted_sd_bino_mono_central.groupby(['eye', 'sd'])['value'].agg(['mean', 'std']).reset_index().sort_values('sd')

print(f'All points \n {sd_summary_bino_mono}')
print(f'Central points \n {sd_summary_bino_mono_cenrtal}')

#%%
# S2S per monocular and binocular with both eye together
melted_s2s = pd.melt(s2s_subject, id_vars=['eye'], value_vars=['precision x S2S', 'precision y S2S'], 
                      var_name='s2s', value_name='value')
melted_s2s['s2s'] = melted_s2s['s2s'].replace({'precision x S2S': 'x', 'precision y S2S': 'y'})

melted_s2s_central = pd.melt(s2s_subject_central, id_vars=['eye'], value_vars=['precision x S2S', 'precision y S2S'], 
                      var_name='s2s', value_name='value')
melted_s2s_central['s2s'] = melted_s2s_central['s2s'].replace({'precision x S2S': 'x', 'precision y S2S': 'y'})

melted_s2s_bino_mono = melted_s2s.copy()
melted_s2s_bino_mono['eye'] = melted_s2s_bino_mono['eye'].replace({'bino left': 'binocular', 'bino right': 'binocular', 'mono left': 'monocular', 'mono right': 'monocular'})

melted_s2s_bino_mono_central = melted_s2s_central.copy()
melted_s2s_bino_mono_central['eye'] = melted_s2s_bino_mono_central['eye'].replace({'bino left': 'binocular', 'bino right': 'binocular', 'mono left': 'monocular', 'mono right': 'monocular'})

figure3, axs3 = plt.subplots(1,2, figsize=(12, 6))
sns.boxplot(x='s2s', y='value', hue='eye', data=melted_s2s_bino_mono,
            ax=axs3[0], showfliers=False)
sns.stripplot(x='s2s', y='value', hue='eye', data=melted_s2s_bino_mono, 
              dodge=True, alpha=0.7, ax=axs3[0], size=10, 
              edgecolor='white', linewidth=2, legend=False)


axs3[0].set_ylim([0, .3])
axs3[0].set_xlabel('')  
axs3[0].set_ylabel('')
axs3[0].legend_.remove()

sns.boxplot(x='s2s', y='value', hue='eye', data=melted_s2s_bino_mono_central,
            ax=axs3[1], showfliers=False)
sns.stripplot(x='s2s', y='value', hue='eye', data=melted_s2s_bino_mono_central, 
              dodge=True, alpha=0.7, ax=axs3[1], size=10, 
              edgecolor='white', linewidth=2, legend=False)
axs3[1].set_ylim([0, .3])
axs3[1].set_xlabel('')
axs3[1].set_ylabel('')

# Simplify legend by removing the word "eye"
handles, labels = axs3[1].get_legend_handles_labels()
new_labels = [label.replace('eye', '').strip() for label in labels]
axs3[1].legend(handles, new_labels, title='', loc='upper left', bbox_to_anchor=(1, 1))

figure3.align_ylabels()

# Set a shared y-label
figure3.text(0.04, 0.5, 'S2S ($^\circ$)', va='center', rotation='vertical', fontsize=12)

plt.show()

s2s_summary_bino_mono = melted_s2s_bino_mono.groupby(['eye', 's2s'])['value'].agg(['mean', 'std']).reset_index().sort_values('s2s')
s2s_summary_bino_mono_cenrtal = melted_s2s_bino_mono_central.groupby(['eye', 's2s'])['value'].agg(['mean', 'std']).reset_index().sort_values('s2s')

print(s2s_summary_bino_mono)
print(s2s_summary_bino_mono_cenrtal)