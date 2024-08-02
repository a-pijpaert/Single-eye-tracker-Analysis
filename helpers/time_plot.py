import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

def time_plot(eyes, 
              gaze_data_per_stimulus,
              data_camera1,
              colors,
              gaze):
    
    num_eyes = len(eyes)
    # Plotting the data
    fig, axs = plt.subplots(num_eyes, 1, figsize=(10, 8), sharex=True)
    for eye_index, eye in enumerate(eyes):
        # print(f'This is the plot for {eye}')
        if num_eyes > 1:
            ax = axs[eye_index]
        else:
            ax = axs

        color_index = 0
        color_counter = 0
        # Draw a horizontal dashed line at timestamps[0] per stimulus for left gaze
        for key, values in gaze_data_per_stimulus[eye].items():
            differences = np.abs(data_camera1['timestamps'] - values['timestamps'][0])
            start_index = np.argmin(differences)
            differences = np.abs(data_camera1['timestamps'] - values['timestamps'][-1])
            end_index = np.argmin(differences)

            ax.axvspan(data_camera1['time'][start_index], 
                        data_camera1['time'][end_index],
                        facecolor=colors[color_index],
                        alpha=0.3)
            
            color_counter += 1
            if color_counter % 5 == 0 and color_counter != 0:
                color_index += 1

        # Left gaze
        ax.plot(data_camera1['time'], gaze[eye]['pog_degrees'][:,0], label='Gaze X', color='r')
        ax.plot(data_camera1['time'], gaze[eye]['pog_degrees'][:,1], label='Gaze Y', color='b')
        ax.set_title(f'{eye} Gaze Data')
        ax.set_ylabel('POG (degrees)')
        ax.set_xlabel('time (s)')
        ax.grid(axis='y', color='0.85')

        legend_patch = mpatches.Patch(color='g', alpha=0.3, label='fixation window')
        line_gaze_x = mlines.Line2D([], [], color='r', label='Gaze X')
        line_gaze_y = mlines.Line2D([], [], color='b', label='Gaze Y')
        ax.legend(handles=[line_gaze_x, 
                            line_gaze_y,
                            legend_patch],
                            loc='upper right')

    plt.tight_layout()
    # plt.show()