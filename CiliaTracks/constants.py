"""
Constants defining required columns and conversion targets for TrackMate CSV data.

- ALL_TRACK_COLUMNS: List of all expected columns in track statistics CSV.
- ALL_SPOTS_COLUMNS: List of all expected columns in spots statistics CSV.
- Track_columns_for_conversion: Subset of track columns that require unit conversion.
- Spots_columns_for_conversion: Subset of spots columns that require unit conversion.
- Feature_Order: Correct feature order for XGBoost prediction
"""


ALL_TRACK_COLUMNS = [
    'TRACK_ID',
    'TRACK_MEAN_QUALITY',
    'TRACK_DISPLACEMENT',
    'TRACK_X_LOCATION',
    'TRACK_Y_LOCATION',
    'TRACK_Z_LOCATION',
    'TRACK_MEAN_SPEED',
    'TRACK_MAX_SPEED',
    'TRACK_MIN_SPEED',
    'TRACK_MEDIAN_SPEED',
    'TRACK_STD_SPEED',
    'TOTAL_DISTANCE_TRAVELED',
    'MAX_DISTANCE_TRAVELED',
    'MEAN_STRAIGHT_LINE_SPEED',
    'LABEL',
    'TRACK_INDEX',
    'NUMBER_SPOTS',
    'NUMBER_GAPS',
    'NUMBER_SPLITS',
    'NUMBER_MERGES',
    'NUMBER_COMPLEX',
    'LONGEST_GAP',
    'TRACK_DURATION',
    'TRACK_START',
    'TRACK_STOP',
    'CONFINEMENT_RATIO',
    'LINEARITY_OF_FORWARD_PROGRESSION',
    'MEAN_DIRECTIONAL_CHANGE_RATE'
]

ALL_SPOTS_COLUMNS = [
    'TRACK_ID', 'FRAME', 'POSITION_T',
    'POSITION_X', 'POSITION_Y', 'POSITION_Z', 'RADIUS'
]

Track_columns_for_conversion = ['TRACK_DISPLACEMENT','TRACK_X_LOCATION','TRACK_Y_LOCATION','TRACK_Z_LOCATION','TRACK_MEAN_SPEED',
        'TRACK_MAX_SPEED','TRACK_MIN_SPEED','TRACK_MEDIAN_SPEED','TRACK_STD_SPEED','TOTAL_DISTANCE_TRAVELED',
        'MAX_DISTANCE_TRAVELED','MEAN_STRAIGHT_LINE_SPEED']

Spots_columns_for_conversion = ['POSITION_X','POSITION_Y','POSITION_Z','RADIUS']

Feature_Order = ['TRACK_DISPLACEMENT', 'TRACK_MEAN_SPEED', 'TRACK_MAX_SPEED', 'TRACK_MIN_SPEED',
            'TOTAL_DISTANCE_TRAVELED', 'MAX_DISTANCE_TRAVELED', 'CONFINEMENT_RATIO',
            'MEAN_STRAIGHT_LINE_SPEED', 'LINEARITY_OF_FORWARD_PROGRESSION',
            'MEAN_DIRECTIONAL_CHANGE_RATE', 'CIRCULAR_VARIANCE', 'PERCENT_IN_DENSEST_90']