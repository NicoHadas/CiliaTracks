"""
prediction_CNN.py

This module uses a pre-trained Convolutional Neural Network (CNN) to
classify samples. It generates a trajectory image from TrackMate data
in memory and uses it as input for the model.
"""

import importlib.resources
import torch
import torch.nn.functional as F
from .model import TrajectoryCNN
from .trajectory_CNN import trajectory_CNN
from .utils import fig_to_tensor

def prediction_CNN(Tracks, Spots, Conversion=None):
    """
    Generates a trajectory image and predicts its class using a pre-trained CNN.

    This function takes TrackMate CSV files, generates a standardized CNN input
    image in memory, and feeds it into the `TrajectoryCNN` model to obtain a
    classification ('Control' or 'PCD'). The result is printed to the console.

    Parameters:
    -----------
    Tracks : str
        Path to the track_statistics CSV file exported from TrackMate.

    Spots : str
        Path to the spots_statistics CSV file exported from TrackMate.

    Conversion : float, optional
        A unit conversion factor to apply to all distance values (e.g., pixel
        to micrometer). If None, no conversion is applied.
    """
    # Generate image
    fig = trajectory_CNN(Tracks, Spots, Conversion)

    # Convert to tensor
    input_tensor = fig_to_tensor(fig)

    # Load CNN model
    with importlib.resources.path('CiliaTracks.models', 'trajectory_cnn_model.pth') as model_path:
        CNN_model = TrajectoryCNN()
        CNN_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        CNN_model.eval()

    # Make prediction 
    with torch.no_grad():
        outputs = CNN_model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
    class_names = ['Control', 'PCD']
    prediction = class_names[predicted_idx.item()]
    confidence_score = confidence.item() * 100 

    # Print prediction
    print(f"Model Prediction: {prediction}")
    print(f"Model Confidence: {confidence_score:.1f}%")  