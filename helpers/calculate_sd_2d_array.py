import numpy as np

def calculate_sd_2d_array(data):
    # Ensure data is a 2D array with two columns
    assert data.shape[1] == 2, "Input data must have two columns for x and y coordinates."
    
    # Filter out rows with NaNs
    valid_data = data[~np.isnan(data).any(axis=1)]

    if valid_data.shape[0] == 0:
        return np.nan  # Return NaN if no valid data remains
    
    x_coords = valid_data[:, 0]
    y_coords = valid_data[:, 1]
    
    # Calculate means of x and y coordinates
    mu_x = np.mean(x_coords)
    mu_y = np.mean(y_coords)
    
    # Calculate squared differences from the mean
    squared_diff_x = (x_coords - mu_x) ** 2
    squared_diff_y = (y_coords - mu_y) ** 2
    
    # Sum of squared differences
    sum_squared_diff = np.sum(squared_diff_x + squared_diff_y)
    
    # Number of valid points
    n = valid_data.shape[0]
    
    # Combined standard deviation
    theta_SD = np.sqrt(sum_squared_diff / n)
    
    return theta_SD

