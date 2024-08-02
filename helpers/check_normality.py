import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def check_normality(data: np.ndarray):
    """
    Check if the provided 1D numerical data is normally distributed.
    
    This function performs a series of visual and statistical tests to assess the normality
    of the provided data. It plots a histogram and Q-Q plot, and performs the Shapiro-Wilk, 
    Kolmogorov-Smirnov, Anderson-Darling, and D’Agostino’s K-squared tests.

    Parameters:
    data (np.ndarray): A 1D numpy array of numerical data.

    Returns:
    None: The function displays plots and prints test results.
    """
    # Ensure data is a numpy array
    data = np.asarray(data)
    
    # Check if the data is 1D
    if data.ndim != 1:
        raise ValueError("Input data must be a 1D numpy array")

    # 1. Visual Inspection

    plt.figure(figsize=(14, 7))

    # Histogram
    plt.subplot(1, 2, 1)
    plt.hist(data, bins=10, edgecolor='black', alpha=0.7)
    plt.title('Histogram of Data')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    # Q-Q Plot
    plt.subplot(1, 2, 2)
    stats.probplot(data, dist="norm", plot=plt)
    plt.title('Q-Q Plot')

    plt.tight_layout()
    plt.show()

    # 2. Statistical Tests

    # Shapiro-Wilk Test
    shapiro_test = stats.shapiro(data)
    print(f'Shapiro-Wilk Test: Statistic={shapiro_test.statistic:.4f}, p-value={shapiro_test.pvalue:.4f}')
