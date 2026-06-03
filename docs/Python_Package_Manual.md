# CiliaTracks  
## Instruction Manual

---

## Table of Contents
1. [Overview](#1-overview)  
2. [Installation](#2-installation)  
   - [2.1 Recommended Installation](#21-recommended-installation)  
   - [2.2 Install from GitHub](#22-install-from-github-directly)  
3. [Generating raw data from Fiji ImageJ](#3-generating-raw-data-from-fiji-imagej-trackmate)  
4. [Python package usage](#4-python-package-usage)  
   - [4.1 Core functions](#41-core-functions)  
     - [4.1.1 converter()](#411-converter)  
     - [4.1.2 trajectory()](#412-trajectory)  
     - [4.1.3 displacement()](#413-displacement)  
     - [4.1.4 speed()](#414-speed)  
     - [4.1.5 track_ML()](#415-track_ml)  
     - [4.1.6 prediction_ML()](#416-prediction_ml)  
     - [4.1.7 trajectory_CNN()](#417-trajectory_cnn)  
     - [4.1.8 prediction_CNN()](#418-prediction_cnn)  
   - [4.2 Model extraction](#42-model-extraction)  
5. [Website usage](#5-website-usage)  
   - [5.1 CiliaTracks Analysis](#51-ciliatracks-analysis)  
   - [5.2 CNN Model](#52-cnn-model)  
   - [5.3 XGBoost Model](#53-xgboost-model)  

---

## 1 Overview

CiliaTracks is a comprehensive Python package and dedicated application that enables clinicians and researchers to analyze cilia-driven particle motility and accurately predict PCD from HVSM data.

The Python package is operating-system independent, while a set of executable applications have been created for Windows, macOS and Linux systems. 

### 1.1 Python Package

The Python package contains the set of visualization, quantitative analysis and prediction computational tools and is easily accessible in the Python framework. The functions are categorized as follows.

- Trajectory Visualization: Generate high-quality angular plots that visualize particle trajectory, displacement, and speed.  
- Feature Engineering for Classification: Automatically extract a comprehensive feature set from track data for input into our XGBoost model for classifying PCD and healthy samples. The feature set could also be used to train and develop new machine learning classifiers.  
- Standardized Image Generation for CNNs: Produce normalized trajectory images suitable as direct input for our Convolutional Neural Network (CNN) model.  
- Direct PCD Classification: Make predictions on whether a sample is from a healthy control or a PCD patient using two pre-trained models: an XGBoost model for feature-based classification and a CNN for image-based classification.  
- Automated Fiji ImageJ Script: The GitHub repository contains a Jython script that automates the entire particle tracking analysis workflow.  

### 1.2 Applications

The corresponding applications provide an accessible and user-friendly alternative for applying the tools/functions in Python. Users can easily upload raw Fiji ImageJ TrackMate data to obtain diagnostic predictions and carry out trajectory analyses.

---

## 2 Installation

The applications can be directly downloaded from Zenodo.

The Python package can be installed in two ways. Please note that you must have Python installed to use the package. Python can be downloaded from their official site for your operating system. 

### 2.1 Recommended Installation

The easiest way to install the package is from the Python Package Index (PyPI). This will provide the latest stable release.

`pip install CiliaTracks`

### 2.2 Install from GitHub directly

To get the latest development version, you can install the package directly from the GitHub repository:

`git+https://github.com/NicoHadas/CiliaTracks.git`

---

## 3 Generating raw data from Fiji ImageJ (TrackMate)

The GitHub repository includes a Jython script that automates the entire TrackMate analysis process. This script will take a folder of raw AVI video files and generate the Tracks.csv and Spots.csv files required for CiliaTracks.

Follow the instructions below to use the script in Fiji ImageJ:

1. Download script: Find the script in the Fiji_Scripts folder of this repository.

<img width="1090" height="191" alt="Screenshot 2026-04-24 at 10 49 24 PM" src="https://github.com/user-attachments/assets/d4d13381-0712-4edf-bc64-31b6a06c23b4" />

2. Open the Script Editor in Fiji: In the main Fiji toolbar, navigate to File > New > Script  
3. Open the Script File: Select the script file you downloaded  
4. Run the Script: Click the "Run" button at the bottom of the editor  

<img width="1089" height="428" alt="Screenshot 2026-04-24 at 10 49 31 PM" src="https://github.com/user-attachments/assets/45a47bb9-21c8-4e7a-9aae-246fcd3564a7" />

5. Select Folders:  
   - Select the input folder containing your .avi video files  
   - Select an output folder where the resulting .csv files will be saved  

6. Click "OK" to start the batch processing. The script will analyze all videos in the input folder and save the results.

Note: If using a confocal microscope, we find it best to modify the script and increase the quality threshold (‘THRESHOLD’) to 3.5 (line 33) and ‘MAX_FRAME_GAP’ to 6 (line 112) of Automated_Trackmate_Analysis.py (CiliaTracks/Fiji_Scipts)

---

## 4 Python package usage

All functions in this package require two CSV files exported from Fiji/ImageJ using the TrackMate plugin:

- Tracks.csv: Contains statistics for each unique track.  
- Spots.csv: Contains information for every detected spot in the video.  

### 4.1 Core functions

This reference guide provides a detailed description of the core functions available in the CiliaTracks package.

#### 4.1.1 converter()

`CiliaTracks.converter(Tracks, Spots, Conversion)`

Applies a unit conversion to TrackMate data and returns the converted data as CSV-formatted strings. This is useful for scaling distance-based measurements from pixels to a physical unit like micrometers.

**Parameters:**
- `Tracks`: str
  - The file path to the track_statistics.csv file exported from TrackMate.
- `Spots`: str
  - The file path to the spots_statistics.csv file exported from TrackMate.
- `Conversion`: float or int (optional)
  - The numerical conversion factor to apply to all distance and speed-related columns in the Tracks and Spots data.

**Returns:**
- `tuple of str`
  - A tuple containing two strings: the first is the converted tracks data in CSV format, and the second is the converted spots data in CSV format.

#### 4.1.2 trajectory()

`CiliaTracks.trajectory(Tracks, Spots, Conversion=None)`

Generates a polar plot visualizing the complete trajectories of particles from TrackMate data. Each track is normalized to a common origin, and the overall plot is centered around the mean angle of all tracks. The top 150 tracks, sorted by quality, are plotted.

**Parameters:**
- `Tracks`: str
  - The file path to the track_statistics.csv file.
- `Spots`: str
  - The file path to the spots_statistics.csv file.
- `Conversion`: float, int, or None, default None
  - A unit conversion factor to apply to distance values. If None, no conversion is applied.

**Returns:**
- `matplotlib.figure.Figure`
  - The figure object for the generated polar trajectory plot, which can be displayed or saved.

#### 4.1.3 displacement()

`CiliaTracks.displacement(Tracks, Spots, Conversion=None)`

Generates a polar displacement plot from TrackMate data. Each track is represented as an arrow from the origin, indicating its net movement direction and distance. The top 150 tracks, sorted by quality, are plotted and centered around the mean angle.

**Parameters:**
- `Tracks`: str
  - The file path to the track_statistics.csv file.
- `Spots`: str
  - The file path to the spots_statistics.csv file.
- `Conversion`: float, int, or None, default None
  - A unit conversion factor to apply to distance values. If None, no conversion is applied.

**Returns:**
- `matplotlib.figure.Figure`
  - The figure object for the generated polar displacement plot.

#### 4.1.4 speed()

`CiliaTracks.speed(Tracks, Spots, Conversion=None)`

Generates a polar plot of track speed vectors from TrackMate data. Each track is represented by an arrow whose length corresponds to the track's mean straight-line speed. The top 150 tracks are processed and centered around the mean angle.

**Parameters:**
- `Tracks`: str
  - The file path to the track_statistics.csv file.
- `Spots`: str
  - The file path to the spots_statistics.csv file.
- `Conversion`: float, int, or None, default None
  - A unit conversion factor to apply to distance values. If None, no conversion is applied.

**Returns:**
- `matplotlib.figure.Figure`
  - The figure object for the generated polar speed vector plot.

#### 4.1.5 track_ML()

`CiliaTracks.track_ML(Tracks, Spots, Conversion=None)`

Extracts key track features from TrackMate data and returns them as a pandas DataFrame. These features are designed for use as input to an XGBoost machine learning model for classifying samples.

**Parameters:**
- `Tracks`: str
  - The file path to the track_statistics.csv file.
- `Spots`: str
  - The file path to the spots_statistics.csv file.
- `Track_Number`: int, default=150
  - Number of tracks to be considered
- `Track_Quality`: bool, default=False
  - If True, enables track-quality tier selection (Top/Mid/Bottom) via Track_Quality_Tier
- `Track_Quality_Tier`: {"Top","Mid","Bottom"}, default="Top"
  - Which quality tier to use when Track_Quality=True:
    - `"Top"`: highest-quality tracks (default behavior)
    - `"Good"`: The next N tracks by quality tracks after top
    - `"Mid"`: middle N tracks by quality rank
    - `"Bottom"`: lowest-quality tracks
- `Conversion`: float, int, or None, default None
  - A unit conversion factor to apply to distance values. If None, no conversion is applied.

**Returns:**
- `pandas.DataFrame`
  - A single-row DataFrame containing the mean feature values for the top 150 tracks.

#### 4.1.6 prediction_ML()

`CiliaTracks.prediction_ML(Tracks, Spots, Conversion=None)`

Extracts features using track_ML and classifies the sample using a pre-trained XGBoost model. The function prints the prediction ('Control' or 'PCD') and the model's confidence score.

**Parameters:**
- `Tracks`: str
  - The file path to the track_statistics.csv file.
- `Spots`: str
  - The file path to the spots_statistics.csv file.
- `Conversion`: float, int, or None, default None
  - A unit conversion factor to apply to distance values. If None, no conversion is applied.

#### 4.1.7 trajectory_CNN()

`CiliaTracks.trajectory_CNN(Tracks, Spots, Conversion=None)`

Generates a standardized polar trajectory plot suitable as input for a Convolutional Neural Network (CNN). The plot is minimalistic, with white trajectories on a black background, and all axes and labels removed.

**Parameters:**
- `Tracks`: str
  - The file path to the track_statistics.csv file.
- `Spots`: str
  - The file path to the spots_statistics.csv file.
- `Conversion`: float, int, or None, default None
  - A unit conversion factor to apply to distance values. If None, no conversion is applied.

**Returns:**
- `matplotlib.figure.Figure`
  - The figure object for the generated CNN input image.

#### 4.1.8 prediction_CNN()

`CiliaTracks.prediction_CNN(Tracks, Spots, Conversion=None)`

Generates a trajectory image in memory and uses a pre-trained Convolutional Neural Network (CNN) to classify the sample. The function prints the prediction ('Control' or 'PCD') and the model's confidence score.

**Parameters:**
- `Tracks`: str
  - The file path to the track_statistics.csv file.
- `Spots`: str
  - The file path to the spots_statistics.csv file.
- `Conversion`: float, int, or None, default None
  - A unit conversion factor to apply to distance values. If None, no conversion is applied.

---

### 4.2 Model extraction

Both the pretrained CNN and XGBoost models are available in the GitHub repository under the Models folder.

---

## 5 Website usage

The corresponding CiliaTracks website has three different tabs:

1. CiliaTracks Analysis  
2. CNN Model  
3. XGBoost Model  

---

### 5.1 CiliaTracks Analysis

Five core, non-predictive CiliaTracks functions (from corresponding the Python package) are hosted in the analysis tab. These functions all accept TrackMate raw data CSVs and an optional conversion factor.

<img width="892" height="572" alt="Screenshot 2026-04-24 at 10 49 51 PM" src="https://github.com/user-attachments/assets/cdbaa3d4-aa28-4c55-931f-9fe35e2c9389" />

1. Trajectory Plot  
2. Displacement Plot  
3. Speed Plot  
4. CNN Input Image  
5. XGBoost Features  

For example, a trajectory plot can be easily generated as shown below.

<img width="667" height="543" alt="Screenshot 2026-04-24 at 10 50 02 PM" src="https://github.com/user-attachments/assets/55e35bb6-86f7-4682-8fdb-c35d1ee0fdaa" />

---

### 5.2 CNN Model

Users can easily generate predictions with TrackMate raw CSVs using our CNN model. Below is the CNN Model tab, where users can input their raw data.

<img width="618" height="473" alt="Screenshot 2026-04-24 at 10 50 08 PM" src="https://github.com/user-attachments/assets/a177647e-267e-4acf-b936-1898d6eee945" />

A subsequent prediction is generated along with the processed trajectory plot that served as input for the CNN model.

<img width="647" height="548" alt="Screenshot 2026-04-24 at 10 50 16 PM" src="https://github.com/user-attachments/assets/07f7a8c9-1ca9-48ab-b429-c6800adcdc94" />

Alternatively, users can also upload processed gray scale trajectory plots directly into the model and generate a prediction.

<img width="710" height="490" alt="Screenshot 2026-04-24 at 10 50 22 PM" src="https://github.com/user-attachments/assets/96362814-29b9-4337-af09-5f19d32ae3eb" />

---

### 5.3 XGBoost Model

For feature-based predictions, users can upload raw TrackMate CSVs to generate a prediction with the XGBoost model. Below is the XGBoost model tab.

<img width="598" height="458" alt="Screenshot 2026-04-24 at 10 50 29 PM" src="https://github.com/user-attachments/assets/6426333d-dbb5-47fd-83f7-d8dcc786b4d6" />

CSV inputs are processed, transformed into a vector of 12 features and then input into the model for a prediction, along with the generated features.

<img width="580" height="545" alt="Screenshot 2026-04-24 at 10 50 39 PM" src="https://github.com/user-attachments/assets/6644346b-cc57-468a-8b61-cf31249be5b9" />

Users also have the option to manually input the 12 predictive track features directly into the model for a prediction.

<img width="668" height="439" alt="Screenshot 2026-04-24 at 10 50 45 PM" src="https://github.com/user-attachments/assets/8239572a-0141-4a3e-af1f-60a8b71e4c1b" />
