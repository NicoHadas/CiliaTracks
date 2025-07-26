"""
track_ml.py

This module provides functionality to extract the key track features from TrackMate output CSV files.
These features can be used directly as input for an XGBoost machine learning model
to classify samples as PCD or Control.

"""

import os 
import pandas as pd
import numpy as np
from .utils import circular_variance_from_angles, percent_densest_90, check_conversion_value
from .constants import ALL_TRACK_COLUMNS, ALL_SPOTS_COLUMNS, Track_columns_for_conversion, Spots_columns_for_conversion

def track_ML(Tracks, Spots, Conversion=None):
    """
    Extracts key track features from TrackMate data as a DataFrame for ML input.
    
    Parameters:
    -----------
    Tracks : str
        Path to track_statistics CSV.
    Spots : str
        Path to spots_statistics CSV.
    Conversion : float or None
        Unit conversion factor (e.g., pixels to micrometers).
    
    Returns:
    --------
    pandas.DataFrame
        Single-row DataFrame with mean feature values for top 150 tracks,
        including circular variance and percent in densest 90° window.
    """
    ## -- Checks --
    # File existence
    if not os.path.isfile(Tracks):
        raise FileNotFoundError(f"Tracks file not found: {Tracks}")
    if not os.path.isfile(Spots):
        raise FileNotFoundError(f"Spots file not found: {Spots}")
    
    # Conversion argument validity
    check_conversion_value(Conversion)

    # Load TrackMate data
    track_stats = pd.read_csv(Tracks, skiprows=[1, 2, 3]) 
    spots_stats = pd.read_csv(Spots, skiprows=[1, 2, 3]) 

    ## -- Checks --
    # Missing required columns
    missing_track_cols = [col for col in ALL_TRACK_COLUMNS if col not in track_stats.columns]
    missing_spots_cols = [col for col in ALL_SPOTS_COLUMNS if col not in spots_stats.columns]
    if missing_track_cols:
        raise ValueError(f"Missing required columns in Tracks CSV: {missing_track_cols}")
    if missing_spots_cols:
        raise ValueError(f"Missing required columns in Spots CSV: {missing_spots_cols}")
    
    ## -- Spots Processing -- 

    # Convert units if needed
    if isinstance(Conversion, (int, float)): 
        spots_stats[Spots_columns_for_conversion] = spots_stats[Spots_columns_for_conversion] * Conversion

    # Keep only first and last Frame for every TRACK_ID in the spots_statistics table
    spots_stats = spots_stats.sort_values(by=['TRACK_ID', 'FRAME'])
    spots_stats = spots_stats.groupby('TRACK_ID', group_keys=False).apply(lambda x: x.iloc[[0, -1]])
    spots_stats = spots_stats.reset_index(drop=True)

    # Split into two new tables: First Frame, Last Frame
    first_frame = spots_stats.drop_duplicates(subset='TRACK_ID', keep='first')
    last_frame = spots_stats.drop_duplicates(subset='TRACK_ID', keep='last')

    ## -- Track Processing --
    
    # Convert units if needed 
    if isinstance(Conversion, (int, float)):
        track_stats[Track_columns_for_conversion] = track_stats[Track_columns_for_conversion] * Conversion

    # Sort by track quality
    track_stats = track_stats.sort_values(by='TRACK_MEAN_QUALITY', ascending=False)

    # Take only the top 150 of tracks 
    track_stats = track_stats[0:150]


    ## -- General Engineering --

    # Merge first and last frame data
    track_stats = track_stats.merge(first_frame[['TRACK_ID', 'POSITION_X', 'POSITION_Y']], on='TRACK_ID')
    track_stats = track_stats.rename(columns={'POSITION_X': 'First_POSITION_X', 'POSITION_Y': 'First_POSITION_Y'})
    track_stats = track_stats.merge(last_frame[['TRACK_ID', 'POSITION_X', 'POSITION_Y']], on='TRACK_ID')
    track_stats = track_stats.rename(columns={'POSITION_X': 'Last_POSITION_X', 'POSITION_Y': 'Last_POSITION_Y'})

    # Calculate displacement and angle
    track_stats['delta_x'] = track_stats['Last_POSITION_X'] - track_stats['First_POSITION_X']
    track_stats['delta_y'] = track_stats['Last_POSITION_Y'] - track_stats['First_POSITION_Y']
    track_stats['angle_radians'] = np.arctan2(track_stats['delta_y'], track_stats['delta_x'])
    track_stats['ANGLE_DEGREES'] = np.degrees(track_stats['angle_radians'])

    # Keep only relevant columns
    cols = ['TRACK_DISPLACEMENT','TRACK_MEAN_SPEED','TRACK_MAX_SPEED','TRACK_MIN_SPEED',
            'TOTAL_DISTANCE_TRAVELED','MAX_DISTANCE_TRAVELED','MEAN_STRAIGHT_LINE_SPEED',
            'CONFINEMENT_RATIO','LINEARITY_OF_FORWARD_PROGRESSION','MEAN_DIRECTIONAL_CHANGE_RATE']

    ## -- Create a DataFrame for the sample --

    mean_series = track_stats[cols].mean()
    df = mean_series.to_frame().T
    df.insert(0, "Sample", "Sample1")


    # Calculate mean displacement
    displacement = np.array(track_stats['TRACK_DISPLACEMENT'])
    d_mean = displacement.mean()

    # Convert angles to radians and keep only angles for tracks with larger than mean displacement
    angles = np.array(track_stats['ANGLE_DEGREES'][displacement >= d_mean])
    angles_rad = np.deg2rad(angles)

    # Add circular variance
    df['CIRCULAR_VARIANCE'] = circular_variance_from_angles(angles_rad)

    # Add Percent in Densest 90 Window
    df["PERCENT_IN_DENSEST_90"] = percent_densest_90(angles)

    print(df)
    return df