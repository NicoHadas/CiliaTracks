import os 
import pandas as pd
import numpy as np
from .constants import ALL_TRACK_COLUMNS, ALL_SPOT_COLUMNS, Track_columns_for_conversion, Spot_columns_for_conversion

def converter(Tracks, Spots, Conversion):
    
    ## -- Checks --
    # File existence
    if not os.path.isfile(Tracks):
        raise FileNotFoundError(f"Tracks file not found: {Tracks}")
    if not os.path.isfile(Spots):
        raise FileNotFoundError(f"Spots file not found: {Spots}")
    
    # Conversion value validity
    if not isinstance(Conversion, (int, float)):
        raise TypeError("Conversion must be a number (int or float).")
    
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

     ## -- Track Converting -- 
    track_stats[Track_columns_for_conversion] = track_stats[Track_columns_for_conversion] * Conversion

    ## -- Spot Converting -- 
    spot_stats[Spot_columns_for_conversion] = spot_stats[Spot_columns_for_conversion] * Conversion

    ## -- Save --

    # Add three blank rows for downstream compatability
    blank_lines = "\n\n\n"
    track_csv = blank_lines + track_stats.to_csv(index=False)
    spot_csv = blank_lines + spot_stats.to_csv(index=False)

    return track_csv, spot_csv