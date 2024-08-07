import numpy as np
import warnings

def calculate_bcea(data: np.ndarray, k: float = 1.14) -> float:
    """
    Calculate the Bivariate Contour Ellipse Area (BCEA) for eye-tracker precision analysis.

    The BCEA quantifies the spread of fixation points in eye-tracking data, providing a metric
    for the precision of an eye-tracker. This function handles NaN values by excluding any pairs 
    of (x, y) where either value is NaN.

    Parameters:
    data (np.ndarray): A 2D numpy array with shape (n, 2) where n is the number of gaze points.
                       The first column should contain the x coordinates and the second column
                       should contain the y coordinates.
    k (float, optional): The scaling factor based on the desired confidence interval.
                         Default is 1.14, which corresponds roughly to a 68.3% confidence interval.

    Returns:
    float: The calculated BCEA value.

    Raises:
    ValueError: If the input array is not 2D with shape (n, 2).
                If no valid data points are available after removing NaNs.

    Example:
    >>> data = np.array([[1, 2], [2, 3], [np.nan, 4], [4, np.nan], [5, 6]])
    >>> calculate_bcea(data)
    BCEA: <calculated_value>
    """
    
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError("Input data must be a 2D numpy array with shape (n, 2)")

    x_coords = data[:, 0]
    y_coords = data[:, 1]

    # Remove NaN values
    valid_mask = ~np.isnan(x_coords) & ~np.isnan(y_coords)
    x_coords = x_coords[valid_mask]
    y_coords = y_coords[valid_mask]

    # Number of valid points
    n = len(x_coords)
    
    if n == 0:
        warnings.warn("No valid data points available after removing NaNs", UserWarning)
        return np.nan, np.nan, np.nan, np.nan
    
    # Standard deviations
    sigma_x = np.std(x_coords, ddof=1)
    sigma_y = np.std(y_coords, ddof=1)
    
    # Covariance
    covariance = np.cov(x_coords, y_coords, ddof=1)
    sigma_xy = covariance[0, 1]

    # Correlation coefficient
    rho = sigma_xy / (sigma_x * sigma_y)
    
    # BCEA calculation
    bcea = 2 * k * np.pi * sigma_x * sigma_y * np.sqrt(1 - rho**2)
    
    return bcea, sigma_x, sigma_y, rho

