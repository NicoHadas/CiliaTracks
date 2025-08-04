"""
prediction_ML.py

This module uses a pre-trained XGBoost model to classify samples.
It extracts key track features from TrackMate data and inputs them into
the model to generate a prediction.
"""

import joblib
import importlib.resources
import os 
import pandas as pd
import numpy as np
import warnings
import sys
from. track_ML import track_ML
from contextlib import redirect_stdout
from .utils import circular_variance_from_angles, percent_densest_90, check_conversion_value
from .constants import ALL_TRACK_COLUMNS, ALL_SPOTS_COLUMNS, Track_columns_for_conversion, Spots_columns_for_conversion, Feature_Order

def prediction_ML(Tracks, Spots, Conversion=None):
    """
    Extracts features and predicts a class using a pre-trained XGBoost model.

    This function processes TrackMate CSV files using the track_ML module to
    generate a feature set. It then loads a pre-trained XGBoost model to
    classify the sample (e.g., 'Control' or 'PCD') and prints the prediction
    and confidence score to the console.

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
    warnings.filterwarnings('ignore', category=UserWarning) 

    # Run track_ML to get predictive features for sample
    with open(os.devnull, 'w') as devnull:
        with redirect_stdout(devnull):
            feature_df = track_ML(Tracks, Spots, Conversion)
    
    input_data = feature_df[Feature_Order].values


    # Load XGBoost model
    with importlib.resources.path('CiliaTracks.models', 'tuned_xgboost_model.joblib') as model_path:
        xgboost_model = joblib.load(model_path)

    # Make Predictions
    prediction_idx = xgboost_model.predict(input_data)[0]
    probabilities = xgboost_model.predict_proba(input_data)[0]
    confidence_score = probabilities[prediction_idx] * 100
    class_names = ['Control', 'PCD']
    prediction = class_names[prediction_idx]

    # Print Prediction
    print(f"Model Prediction: {prediction}")
    print(f"Model Confidence: {confidence_score:.1f}%")

