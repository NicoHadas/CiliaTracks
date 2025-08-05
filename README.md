# CiliaTracks

[![PyPI version](https://badge.fury.io/py/CiliaTracks.svg)](https://badge.fury.io/py/CiliaTracks)
[![Python Version](https://img.shields.io/pypi/pyversions/CiliaTracks.svg)](https://pypi.org/project/CiliaTracks)

A set of computational tools for the visualization and quantitative analysis of cilia-driven particle movement, designed to process TrackMate data from high-speed video microscopy of cultured airway cells.

This package provides the functions to analyze and visualize data from TrackMate exports, complementing the research detailed in our publication on Primary Ciliary Dyskinesia (PCD).

---

### Core Capabilities

* **Trajectory Visualization:** Generate high-quality angular plots that visualize particle trajectory, displacement, and speed.

* **Feature Engineering for Classification:** Automatically extract a comprehensive feature set from track data for input into our XGBoost model for classifying PCD and healthy samples. The feature set could also be used to train and develop new machine learning classifiers. 

* **Standardized Image Generation for CNNs:** Produce normalized trajectory images suitable as direct input for our Convolutional Neural Network (CNN) model.

* **Direct PCD Classification:** Make predictions on whether a sample is from a healthy control or a PCD patient using two pre-trained models: an XGBoost model for feature-based classification and a CNN for image-based classification.

* **Automated Fiji ImageJ Script:** This repostiory contains a Jython script that automates the entire particle tracking analysis workflow 
---


### Installation

There are two ways to install `CiliaTracks`:

**Option 1: Install from PyPI (Recommended)**

The easiest way to install the package is from the Python Package Index (PyPI). This will provide the latest stable release.

```bash
pip install CiliaTracks
```

**Option 2: Install from GitHub**

To get the latest development version, you can install the package directly from this GitHub repository.

```bash
pip install git+[https://github.com/NicoHadas/CiliaTracks.git](https://github.com/NicoHadas/CiliaTracks.git)
```

---


### Quick Start

The following example demonstrates how to generate and display a trajectory plot from TrackMate CSV files.

```python
import CiliaTracks as CT
import matplotlib.pyplot as plt

# Define paths to your TrackMate CSV files
tracks_file = 'path/to/your/Tracks.csv'
spots_file = 'path/to/your/Spots.csv'

# Optional: Define a pixel-to-micrometer conversion factor
conversion_factor = 0.65 # e.g., 1 pixel = 0.65 µm

# Generate the trajectory plot figure
fig = CT.trajectory(
    Tracks=tracks_file, 
    Spots=spots_file, 
    Conversion=conversion_factor
)

# Display the plot interactively
plt.show()

# To save the figure
# fig.savefig('trajectory_plot.png', dpi=300, bbox_inches='tight')
```

---

### Prediction Examples

The package includes pre-trained models to classify samples directly from your TrackMate data.

#### 1. XGBoost Model Prediction

```python
import CiliaTracks as CT

# Define paths to your TrackMate CSV files
tracks_file = 'path/to/your/Tracks.csv'
spots_file = 'path/to/your/Spots.csv'
conversion_factor = 0.65

# Get a prediction from the XGBoost model
CT.prediction_ML(
    Tracks=tracks_file,
    Spots=spots_file,
    Conversion=conversion_factor
)
# Expected output:
# Model Prediction: PCD
# Model Confidence: 98.7%
```
#### 2. CNN Model Prediction

```python
import CiliaTracks as CT

# Define paths to your TrackMate CSV files
tracks_file = 'path/to/your/Tracks.csv'
spots_file = 'path/to/your/Spots.csv'
conversion_factor = 0.65

# Get a prediction from the CNN model
CT.prediction_CNN(
    Tracks=tracks_file,
    Spots=spots_file,
    Conversion=conversion_factor
)
# Expected output:
# Model Prediction: Control
# Model Confidence: 95.2%
```
---

### Example Ouputs

<table border="0" cellspacing="0" cellpadding="10" align="center">
  <tr>
    <td align="center">
      <img src="assets/Trajectory.png" alt="Trajectory Plot" width="350">
      <br>
      <b>Fig 1.</b> Trajectory Plot
    </td>
    <td align="center">
      <img src="assets/Displacement.png" alt="Displacement Plot" width="350">
      <br>
      <b>Fig 2.</b> Displacement Plot
    </td>
    <td align="center">
      <img src="assets/CNN.png" alt="CNN Input Plot" width="350">
      <br>
      <b>Fig 3.</b> CNN Input Image
    </td>
  </tr>
</table>

---

### Automated Fiji/ImageJ Analysis Script

This repository includes a Jython script that automates the entire TrackMate analysis process. This script will take a folder of raw AVI video files and generate the Tracks.csv and Spots.csv files required by the CiliaTracks Python package.

#### How to Use the Script

1. Download script: Find the script in the Fiji_Scripts folder of this repository. 
2. Open the Script Editor in Fiji: In the main Fiji toolbar, navigate to File > New > Script
3. Open the Script File: Select the script file you downloaded
4. Run the Script: Click the "Run" button at the bottom of the editor
5. Select Folders: 

    a. Select the input folder containing your .avi video files

    b. Select an output folder where the resulting .csv files will be saved.

6. Click "OK" to start the batch processing. The script will analyze all videos in the input folder and save the results.



### How to Cite

If you use `CiliaTracks` in your research, please cite our publication:
