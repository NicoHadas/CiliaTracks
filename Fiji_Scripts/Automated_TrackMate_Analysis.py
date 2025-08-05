"""
Automated Batch TrackMate Analysis for CiliaTracks

This Jython script automates the entire TrackMate analysis workflow for a
folder of AVI video files. It is designed to generate the specific track
and spot statistics CSV files required as input for the CiliaTracks Python package.

For each AVI file in a specified input directory, the script will:
1.  Open the video and convert it to 8-bit grayscale.
2.  Run the complete TrackMate analysis using pre-defined settings
    (LogDetector for spots, KalmanTracker for tracks).
3.  Save the 'track_statistics' and 'spot_statistics' as CSV files in a
    specified output directory.

--------------------------------------------------------------------------
                          *** HOW TO USE ***
1. OPEN IN FIJI: Open Fiji, then go to File > New > Script... and open this file.
2. RUN: Click the "Run" button in the script editor.
3. SELECT FOLDERS: In the pop-up dialog, select the input folder
   containing your AVI files and an output folder for the CSV results.
4. CLICK OK: The script will process all videos automatically.
--------------------------------------------------------------------------
"""


# Import necessary ImageJ and Fiji libraries
import sys
import os

#@ File (label="Select Input Directory with AVI files", style="directory") input_dir_file
#@ File (label="Select Output Directory for CSV results", style="directory") output_dir_file


from ij import IJ
from fiji.plugin.trackmate import Model, Settings, TrackMate, Logger
from fiji.plugin.trackmate.detection import LogDetectorFactory
from fiji.plugin.trackmate.tracking.kalman import KalmanTrackerFactory
from fiji.plugin.trackmate.gui.displaysettings import DisplaySettingsIO
from fiji.plugin.trackmate.gui.displaysettings.DisplaySettings import TrackMateObject
from fiji.plugin.trackmate.features.track import TrackIndexAnalyzer
import fiji.plugin.trackmate.visualization.hyperstack.HyperStackDisplayer as HyperStackDisplayer
import fiji.plugin.trackmate.features.FeatureFilter as FeatureFilter
from fiji.plugin.trackmate import SelectionModel 

import java.lang.System as System
from java.io import File
from fiji.plugin.trackmate.visualization.table import TrackTableView


# -------------------------
# File set-up
# -------------------------

# Avoid errors with UTF8 chars
reload(sys)
sys.setdefaultencoding('utf-8')

# Define input and output directories
input_directory = str(input_dir_file)
output_directory = str(output_dir_file)

# Loop through all .avi files in the input directory
for filename in os.listdir(input_directory):
    if filename.endswith(".avi"):
        avi_file_path = os.path.join(input_directory, filename)
        print("\nProcessing file: {}".format(filename))

        # Import saved .AVI file into ImageJ
        imp = IJ.openImage(avi_file_path)
        if imp is None:
            print("Could not open {}, skipping...".format(filename))
            continue

        # Convert to grayscale (8-bit)
        IJ.run(imp, "8-bit", "")

        # Flip Z and T dimensions
        dims = imp.getDimensions()
        imp.setDimensions(dims[2], dims[4], dims[3])
        imp.show()

        # -------------------------
        # Create the model object
        # -------------------------

        model = Model()
        model.setLogger(Logger.IJ_LOGGER)

        # ------------------------
        # Prepare settings object
        # ------------------------

        settings = Settings(imp)
        settings.detectorFactory = LogDetectorFactory()
        settings.detectorSettings = {
            'DO_SUBPIXEL_LOCALIZATION': True,
            'RADIUS': 3.5,
            'TARGET_CHANNEL': 1,
            'THRESHOLD': 1.0,
            'DO_MEDIAN_FILTERING': False,
        }

        # Configure spot filters
        filter1 = FeatureFilter('QUALITY', 0, True)
        settings.addSpotFilter(filter1)

        # Configure tracker
        settings.trackerFactory = KalmanTrackerFactory()
        settings.trackerSettings = {
            'LINKING_MAX_DISTANCE': 15.0,
            'KALMAN_SEARCH_RADIUS': 20.0,
            'MAX_FRAME_GAP': 1,
        }

        # Add ALL the feature analyzers
        settings.addAllAnalyzers()

        # -------------------
        # Instantiate TrackMate
        # -------------------

        trackmate = TrackMate(model, settings)

        # --------
        # Process
        # --------

        ok = trackmate.checkInput()
        if not ok:
            print("❌ Error checking input for {}: {}".format(filename, trackmate.getErrorMessage()))
            imp.close()
            continue

        ok = trackmate.process()
        if not ok:
            print("❌ Error processing {}: {}".format(filename, trackmate.getErrorMessage()))
            imp.close()
            continue

        # ----------------
        # Display results
        # ----------------

        selectionModel = SelectionModel(model)  
        ds = DisplaySettingsIO.readUserDefault()
        ds.setTrackColorBy(TrackMateObject.TRACKS, TrackIndexAnalyzer.TRACK_INDEX)
        ds.setSpotColorBy(TrackMateObject.TRACKS, TrackIndexAnalyzer.TRACK_INDEX)

        displayer = HyperStackDisplayer(model, selectionModel, imp, ds)
        displayer.render()
        displayer.refresh()

        # -------------------------
        # Save output files per file
        # -------------------------

        base_name = os.path.splitext(filename)[0]
        trackFile = File(output_directory, "track_statistics_{}.csv".format(base_name))
        spotFile = File(output_directory, "spot_statistics_{}.csv".format(base_name))

        trackTableView = TrackTableView(model, selectionModel, ds, "Track Statistics")
        trackTableView.getTrackTable().exportToCsv(trackFile)
        trackTableView.getSpotTable().exportToCsv(spotFile)

        print("Processing complete for {}, results saved.".format(filename))

        # Close the image to prevent memory leaks
        imp.changes = False
        imp.close()

print("\nAll files processed successfully.")