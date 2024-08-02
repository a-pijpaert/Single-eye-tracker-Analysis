import numpy as np

def calculate_rms_s2s(data: np.ndarray):
    if ~np.isnan(data).all():
        valid_data = np.array([d for d in data if ~np.isnan(d).any()])
        rms_s2s_array = []
        for i in range(valid_data.shape[1]):
            column_data = valid_data[:, i]
            intersample_difference = np.array([np.linalg.norm(column_data[i+1]-column_data[i]) for i in range(len(column_data)-1)])
            sum_intersample_difference_squared = np.sum(intersample_difference**2)
            rms_s2s = np.sqrt(sum_intersample_difference_squared/len(intersample_difference))
            rms_s2s_array.append(rms_s2s)

        # calculate vector length rms s2s 
        intersample_difference = np.array([np.linalg.norm(valid_data[i+1] - valid_data[i]) for i in range(len(valid_data[:,0])-1)])
        sum_intersample_difference_squared = np.sum(intersample_difference**2)
        rms_s2s = np.sqrt(sum_intersample_difference_squared/len(intersample_difference))
        rms_s2s_array.append(rms_s2s)
        
        return np.array(rms_s2s_array)
    
    return np.array([np.nan, np.nan, np.nan])