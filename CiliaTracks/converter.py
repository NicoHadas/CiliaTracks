import os 
import pandas as pd
import numpy as np
from .constants import ALL_TRACK_COLUMNS, ALL_SPOTS_COLUMNS, Track_columns_for_conversion, Spots_columns_for_conversion

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

     ## -- Track Converting -- 
    track_stats[Track_columns_for_conversion] = track_stats[Track_columns_for_conversion] * Conversion

    ## -- Spots Converting -- 
    spots_stats[Spots_columns_for_conversion] = spots_stats[Spots_columns_for_conversion] * Conversion

    ## -- Save --

    # Add three blank rows for downstream compatability
    blank_lines = "\n\n\n"
    track_csv = blank_lines + track_stats.to_csv(index=False)
    spots_csv = blank_lines + spots_stats.to_csv(index=False)

    return track_csv, spots_csv
