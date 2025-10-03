"""
Utility functions for CiliaTracks
"""

import numpy as np
from io import BytesIO
from PIL import Image
from torchvision import transforms
import torch

def circular_variance_from_angles(angles_rad):
    """
    Calculate the circular variance of angles given in radians.

    Circular variance measures the dispersion of angles on the unit circle.
    Values range from 0 (no variance, all angles identical) to 1 (maximum variance).

    Parameters
    ----------
    angles_rad : array-like
        Array or list of angles in radians.

    Returns
    -------
    float
        Circular variance value between 0 and 1.
    """
    mean_cos = np.mean(np.cos(angles_rad))
    mean_sin = np.mean(np.sin(angles_rad))
    R = np.sqrt(mean_cos**2 + mean_sin**2)
    return 1 - R


def mean_angle(angles_rad):
    """
    Compute the mean angle (average direction) of a list of angles in radians.

    Parameters
    ----------
    angles_rad : array-like
        Array or list of angles in radians.

    Returns
    -------
    float
        Mean angle in radians, between -π and π.
    """
    mean_cos_theta = np.mean(np.cos(angles_rad))
    mean_sin_theta = np.mean(np.sin(angles_rad))
    
    return np.arctan2(mean_sin_theta, mean_cos_theta)


def percent_densest_90(angles2):
    """
    Calculate the percentage of angles falling within the densest 90-degree arc.

    This finds the 90° window containing the most angles (converted to [0,360) degrees),
    then returns the percent of total angles in that window.

    Parameters
    ----------
    angles2 : array-like
        Array or list of angles in degrees.

    Returns
    -------
    float
        Percentage (0-100) of angles within the densest 90° arc.
    """
    sorted_angles = np.sort((angles2 + 180) % 360)
    best_count = 0
    for i in range(len(sorted_angles)):
        start = sorted_angles[i]
        end = (start + 90) % 360
        if end > start:
            count = np.sum((sorted_angles >= start) & (sorted_angles <= end))
        else:
            count = np.sum((sorted_angles >= start) | (sorted_angles <= end))  
        best_count = max(best_count, count)

    return best_count / len(angles2) * 100


def is_numeric(val):
    """
    Check if a value is numeric (int or float).

    Parameters
    ----------
    val : any
        Value to check.

    Returns
    -------
    bool
        True if val is an int or float, False otherwise.
    """
    return isinstance(val, (int, float))


def check_conversion_value(Conversion):
    """
    Validate that the conversion value is either numeric or None.

    Raises a TypeError if Conversion is not None and not numeric.

    Parameters
    ----------
    Conversion : int, float, or None
        Value to validate.

    Raises
    ------
    TypeError
        If Conversion is not a number (int or float) and not None.
    """
    if Conversion is not None and not is_numeric(Conversion):
        raise TypeError("Conversion must be a number (int or float) or None.")
    
def fig_to_tensor(fig):
    """
    Converts a Matplotlib figure to a pre-processed PyTorch tensor.

    This function takes a figure, saves it to an in-memory buffer,
    and processes it into the correct tensor format for CNN input.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to convert.

    Returns
    -------
    torch.Tensor
        The pre-processed image tensor.
    """
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    image = Image.open(buf).convert('L')
    preprocess = transforms.Compose([
        transforms.Resize((500, 500)),
        transforms.ToTensor()
    ])
    tensor = preprocess(image).unsqueeze(0)
    buf.close()
    
    return tensor

    

