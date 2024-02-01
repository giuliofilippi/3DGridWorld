# imports
import numpy as np

# entropy with smoothing to avoid log(0).
def entropy(probabilities, epsilon=1e-15):
    """
    Compute entropy of a probability distribution with additive smoothing.

    Parameters:
    - probabilities (numpy array): Probability distribution.
    - epsilon (float): Small value to avoid logarithm of zero.

    Returns:
    - Entropy
    """
    probabilities = np.clip(probabilities, epsilon, 1 - epsilon)
    return -np.sum(probabilities * np.log2(probabilities))

# cross entropy with smoothing to avoid log(0).
def cross_entropy(p_emp, p_stat, epsilon=1e-15):
    """
    Compute cross entropy between true labels and predicted labels with additive smoothing.
    Never used.

    Parameters:
    - p_emp: true probability distribution (as a numpy array)
    - p_stat: predicted probability distribution (as a numpy array)
    - epsilon: small value to avoid logarithm of zero

    Returns:
    - Cross entropy
    """
    p_emp = np.clip(p_emp, epsilon, 1 - epsilon)  # Clip probabilities to avoid log(0)
    p_stat = np.clip(p_stat, epsilon, 1 - epsilon)  # Clip probabilities to avoid log(0)
    return -np.sum(p_emp * np.log(p_stat))