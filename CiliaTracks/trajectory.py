"""
trajectory.py

This module computes trajectory angle plots for cilia-propelled particles.
It processes spot and track statistics and outputs polar angle plots.

"""

import matplotlib.pyplot as plt
import os 
import pandas as pd
import numpy as np
import statistics
from .utils import circular_variance_from_angles, mean_angle, percent_densest_90, check_conversion_value
from .constants import ALL_TRACK_COLUMNS, ALL_SPOT_COLUMNS, Track_columns_for_conversion, Spot_columns_for_conversion


def trajectory(Tracks, Spots, Conversion=None):
    """
    Generates a polar plot of trajectories from TrackMate data. Each track is normalized
    to start at an origin and overall track angles are centered around 0 degrees. Top 150 
    tracks are plotted based on quality. The plot is annotated with statistics 
    (Percent in Densest 90 and Circular Variance).

    Parameters:
    -----------
    Tracks : str
        Path to the track_statistics CSV file exported from TrackMate.

    Spots : str
        Path to the spot_statistics CSV file exported from TrackMate.

    Conversion : float, optional
        A unit conversion factor to apply to all distance values (e.g., pixel to micrometer).
        If None, no conversion is applied.

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object for the generated polar trajectory plot.
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

    # Add Percent in Densest 90 Window
    percent_within_90 = percent_densest_90(angles2)

    # Add circular variance 
    circular_variance = circular_variance_from_angles(angles_rad)

    # Calulate mean angle for circular data
    rad_mean = mean_angle(angles_rad)


    ## -- Visualization --
    # Set up the plot
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})

    # Set the background color of plot to black
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    # Initialize the variable to track the maximum distance
    max_distance = 0

    # Loop through each unique track and plot its trajectory
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
            
            # Update max distance
            if distance > max_distance:
                max_distance = distance
        
        ax.plot(angles, distances, label=f"Track {track_id}")

    # Annotate  plot with the Percent in Densest 90 Window 
    ax.text(
        1.1,  
        -0.025,  
        f"Percent in Densest 90: {percent_within_90:.1f}%", 
        color='white',            
        fontsize=16,
        ha='right', va='bottom',    
        transform=ax.transAxes    
    )

    # Annotate plot with Circular Variance  
    ax.text(
        1.1, 
        -0.075,  
        f"Circular Variance: {circular_variance:.2f}",  
        color='white',            
        fontsize=16,
        ha='right', va='bottom',   
        transform=ax.transAxes    
    )


    # Set the maximum radius to fit all data
    ax.set_rmax(track_stats['TRACK_DISPLACEMENT'].max() + 50)  
    ax.set_rlabel_position(80)

    # Set the title and labels for the polar plot
    ax.set_title("Track Trajectory", color='white', fontsize = 20)

    # Set  grid and axis labels to white 
    ax.grid(True, color='white', linestyle='--', linewidth=0.5)
    ax.tick_params(axis='both', labelcolor='white')

    # Show the plot
    plt.show()
    return fig



