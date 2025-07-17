"""
trajectory_CNN.py

This module generates polar plots of ciliary bead trajectories suitable as input
for CNN predictions to classify primary ciliary dyskinesia (PCD) vs control samples.
It processes TrackMate spot and track statistics, normalizes trajectories relative
to the mean angle, and visualizes them on a black polar plot without axes or grid.
"""

import matplotlib.pyplot as plt
import os 
import pandas as pd
import numpy as np
import statistics
from .utils import mean_angle, check_conversion_value
from .constants import ALL_TRACK_COLUMNS, ALL_SPOT_COLUMNS, Track_columns_for_conversion, Spot_columns_for_conversion


def trajectory_CNN(Tracks, Spots, Conversion=None):
    """
    Generate polar trajectory plots from TrackMate data for CNN input.

    Parameters
    ----------
    Tracks : str
        Path to the TrackMate track_statistics CSV file.
    Spots : str
        Path to the TrackMate spot_statistics CSV file.
    Conversion : float or None, optional
        Factor to convert units (e.g., pixels to micrometers).
        If None, no conversion is applied.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Matplotlib figure object containing the polar trajectory plot.
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
    spot_stats = pd.read_csv(Tracks, skiprows=[1, 2, 3]) 
    track_stats = pd.read_csv(Spots, skiprows=[1, 2, 3]) 

    ## -- Checks --
    # Missing required columns
    missing_track_cols = [col for col in ALL_TRACK_COLUMNS if col not in track_stats.columns]
    missing_spots_cols = [col for col in ALL_SPOT_COLUMNS if col not in spot_stats.columns]
    if missing_track_cols:
        raise ValueError(f"Missing required columns in Tracks CSV: {missing_track_cols}")
    if missing_spots_cols:
        raise ValueError(f"Missing required columns in Spots CSV: {missing_spots_cols}")

     ## -- Track Processing -- 

    # Convert units if needed 
    if isinstance(Conversion, (int, float)):
        track_stats[Track_columns_for_conversion] = track_stats[Track_columns_for_conversion] * Conversion
  
    # Sort by track quality
    track_stats = track_stats.sort_values(by='TRACK_MEAN_QUALITY', ascending=False)
    track_stats = track_stats[0:150]
    filtered_trackids = track_stats['TRACK_ID'].unique().tolist()

    # Duplicate
    track_stats2 = track_stats.copy()

    ## -- Spot Processing -- 

    # Convert units if needed
    if isinstance(Conversion, (int, float)): 
        spot_stats[Spot_columns_for_conversion] = spot_stats[Spot_columns_for_conversion] * Conversion

    # Duplicate
    spot_stats2 = spot_stats.copy()

    # Keep only top tracks
    spot_stats = spot_stats[spot_stats['TRACK_ID'].isin(filtered_trackids)]

    ## -- General Engineering --

    # Keep only first and last Frame for every TRACK_ID in the spot_statistics table
    spot_stats2 = spot_stats2.sort_values(by=['TRACK_ID', 'FRAME'])
    spot_stats2 = spot_stats2.groupby('TRACK_ID').apply(lambda x: x.iloc[[0, -1]])
    spot_stats2 = spot_stats2.reset_index(drop=True)

    # Split into two new tables: First Frame, Last Frame
    First_FRAME = spot_stats2.drop_duplicates(subset='TRACK_ID', keep='first')
    Last_FRAME = spot_stats2.drop_duplicates(subset='TRACK_ID', keep='last')

    # Merge the First and Last FRAME tables to the track statistics table
    track_stats2 = track_stats2.merge(First_FRAME[['TRACK_ID', 'POSITION_X','POSITION_Y']], on='TRACK_ID')
    track_stats2 = track_stats2.rename(columns={'POSITION_X': 'First_POSITION_X', 'POSITION_Y': 'First_POSITION_Y'})
    track_stats2 = track_stats2.merge(Last_FRAME[['TRACK_ID', 'POSITION_X','POSITION_Y']], on='TRACK_ID')
    track_stats2 = track_stats2.rename(columns={'POSITION_X': 'Last_POSITION_X', 'POSITION_Y': 'Last_POSITION_Y'})

    # Calculate the change in positions (delta_x and delta_y)
    track_stats2['delta_x'] = track_stats2['Last_POSITION_X'] - track_stats2['First_POSITION_X']
    track_stats2['delta_y'] = track_stats2['Last_POSITION_Y'] - track_stats2['First_POSITION_Y']

    # Calculate the angle in radians using atan2
    track_stats2['angle_radians'] = np.arctan2(track_stats2['delta_y'], track_stats2['delta_x'])

    # Convert to degrees
    track_stats2['ANGLE_DEGREES'] = np.degrees(track_stats2['angle_radians'])

    # Calculate mean displacement
    displacement = np.array(track_stats2['TRACK_DISPLACEMENT'])
    d_mean = displacement.mean()

    # Convert angles to radians, keeping only angles for tracks with greater than mean displacement
    angles2 = np.array(track_stats2['ANGLE_DEGREES'])[displacement >= d_mean]
    angles_rad = np.deg2rad(angles2)

    # Calulate mean angle for circular data
    rad_mean = mean_angle(angles_rad)

    ## -- Visualization --

    # Setup polar plot
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    # Plot each track centered around mean
    for idx, track_id in enumerate(spot_stats['TRACK_ID'].unique()):
        # Filter the data for current track
        track_data = spot_stats[spot_stats['TRACK_ID'] == track_id].sort_values(by='POSITION_T')  
        
        # Get first point of the track to use as origin
        x0, y0 = track_data.iloc[0][['POSITION_X', 'POSITION_Y']]
        
        # Initialize arrays
        angles = []
        distances = []
        
        # Loop through each point in track
        for i in range(len(track_data)):
            x, y = track_data.iloc[i][['POSITION_X', 'POSITION_Y']]
            
            # Subtract first point to start track from (0, 0)
            x_rel = x - x0
            y_rel = y - y0
            
            # Convert (x, y) to polar coordinates (distance, angle)
            distance = np.sqrt(x_rel**2 + y_rel**2) 
            angle = np.arctan2(y_rel, x_rel)  
            
            # Append calculated values to the lists
            angles.append(angle - rad_mean)
            distances.append(distance)

        # Plot
        ax.plot(angles, distances, color='white', linewidth=0.8)

    # Clean up polar plot
    ax.set_rmax(410)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.show()
    return fig