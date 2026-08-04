import sys
import os
import tempfile
import matplotlib
matplotlib.use('Agg')  # Prevents crashes on macOS/Linux servers
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
from torchvision import transforms
import joblib
import numpy as np
import CiliaTracks as CT
from io import BytesIO
import base64
import webbrowser
from threading import Timer
import xgboost  # Keeps the dependency visible to PyInstaller

# --- Import CNN model architecture ---
from model import TrajectoryCNN


# --- Shared model settings ---
DEVICE = torch.device("cpu")
CLASS_NAMES = ["Control", "PCD"]
FEATURE_ORDER = [
    "TRACK_DISPLACEMENT",
    "TRACK_MEAN_SPEED",
    "TRACK_MAX_SPEED",
    "TRACK_MIN_SPEED",
    "TOTAL_DISTANCE_TRAVELED",
    "MAX_DISTANCE_TRAVELED",
    "CONFINEMENT_RATIO",
    "MEAN_STRAIGHT_LINE_SPEED",
    "LINEARITY_OF_FORWARD_PROGRESSION",
    "MEAN_DIRECTIONAL_CHANGE_RATE",
    "CIRCULAR_VARIANCE",
    "PERCENT_IN_DENSEST_90",
]


FEATURE_LABELS = {
    "TRACK_DISPLACEMENT": "Track displacement",
    "TRACK_MEAN_SPEED": "Track mean speed",
    "TRACK_MAX_SPEED": "Track max speed",
    "TRACK_MIN_SPEED": "Track min speed",
    "TOTAL_DISTANCE_TRAVELED": "Total distance traveled",
    "MAX_DISTANCE_TRAVELED": "Max distance traveled",
    "CONFINEMENT_RATIO": "Confinement ratio",
    "MEAN_STRAIGHT_LINE_SPEED": "Mean straight line speed",
    "LINEARITY_OF_FORWARD_PROGRESSION": "Linearity of forward progression",
    "MEAN_DIRECTIONAL_CHANGE_RATE": "Mean directional change rate",
    "CIRCULAR_VARIANCE": "Circular variance",
    "PERCENT_IN_DENSEST_90": "Percent in densest 90°",
}

FEATURE_UNITS = {
    "TRACK_DISPLACEMENT": "µm",
    "TRACK_MEAN_SPEED": "µm/frame",
    "TRACK_MAX_SPEED": "µm/frame",
    "TRACK_MIN_SPEED": "µm/frame",
    "TOTAL_DISTANCE_TRAVELED": "µm",
    "MAX_DISTANCE_TRAVELED": "µm",
    "CONFINEMENT_RATIO": "Unitless",
    "MEAN_STRAIGHT_LINE_SPEED": "µm/frame",
    "LINEARITY_OF_FORWARD_PROGRESSION": "Unitless",
    "MEAN_DIRECTIONAL_CHANGE_RATE": "radians/frame",
    "CIRCULAR_VARIANCE": "Unitless",
    "PERCENT_IN_DENSEST_90": "%",
}



def build_display_features(feature_values):
    """Return feature labels, formatted values, and units for display tables."""
    return [
        (
            FEATURE_LABELS[feature],
            f"{float(feature_values[feature]):.4f}",
            FEATURE_UNITS[feature],
        )
        for feature in FEATURE_ORDER
    ]


# --- Helper function for PyInstaller (to find data files) ---
def resource_path(relative_path):
    """Return an absolute resource path in development and PyInstaller builds."""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


# --- Basic Flask App Setup ---
app = Flask(
    __name__,
    template_folder=resource_path("templates"),
    static_folder=resource_path("static"),
)

try:
    user_home = os.path.expanduser("~")
    upload_folder = os.path.join(user_home, "CiliaTracksApp_Uploads")
    os.makedirs(upload_folder, exist_ok=True)
except Exception:
    upload_folder = "uploads"
    os.makedirs(upload_folder, exist_ok=True)

app.config["UPLOAD_FOLDER"] = upload_folder


# --- Load Models ---
cnn_path = resource_path("trajectory_cnn_model.pth")
cnn_model = TrajectoryCNN().to(DEVICE)
cnn_model.load_state_dict(torch.load(cnn_path, map_location=DEVICE))
cnn_model.eval()

xgb_path = resource_path("tuned_xgboost_model.joblib")
xgboost_model = joblib.load(xgb_path)


# --- General helpers ---
def preprocess_image(image_path: str):
    image = Image.open(image_path).convert("L")
    preprocess = transforms.Compose([
        transforms.Resize((500, 500)),
        transforms.ToTensor(),
    ])
    image_tensor = preprocess(image)
    return image_tensor.unsqueeze(0)


def fig_to_base64_img(fig):
    """Convert a Matplotlib figure to an embeddable PNG data URL."""
    buf = BytesIO()
    fig.savefig(buf, format="PNG", bbox_inches="tight")
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    buf.close()
    return f"data:image/png;base64,{img_str}"


def fig_to_tensor(fig):
    """Convert a Matplotlib figure into the 500 x 500 grayscale CNN tensor."""
    buf = BytesIO()
    fig.savefig(buf, format="PNG", bbox_inches="tight")
    buf.seek(0)
    with Image.open(buf) as image:
        image = image.convert("L")
        preprocess = transforms.Compose([
            transforms.Resize((500, 500)),
            transforms.ToTensor(),
        ])
        tensor = preprocess(image).unsqueeze(0)
    buf.close()
    return tensor


def run_cnn_prediction(input_tensor):
    """Return the CNN class prediction and confidence percentage."""
    with torch.no_grad():
        outputs = cnn_model(input_tensor.to(DEVICE))
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, dim=1)

    predicted_class = int(predicted_idx.item())
    return CLASS_NAMES[predicted_class], confidence.item() * 100


def run_xgboost_prediction(input_data):
    """Return the XGBoost class prediction and confidence percentage."""
    predicted_label = xgboost_model.predict(input_data)[0]
    probabilities = xgboost_model.predict_proba(input_data)[0]

    model_classes = list(xgboost_model.classes_)
    probability_index = model_classes.index(predicted_label)
    confidence = float(probabilities[probability_index]) * 100

    predicted_class = int(predicted_label)
    return CLASS_NAMES[predicted_class], confidence


def get_uploaded_trackmate_files():
    """Validate and return the uploaded Tracks and Spots file objects."""
    if "tracks_file" not in request.files or "spots_file" not in request.files:
        raise ValueError("Please upload both Tracks and Spots CSV files.")

    tracks_file = request.files["tracks_file"]
    spots_file = request.files["spots_file"]

    if not tracks_file.filename or not spots_file.filename:
        raise ValueError("Please select both files.")

    return tracks_file, spots_file


def parse_conversion_factor():
    """Return the optional pixel conversion factor (µm/pixel) as a float."""
    conversion = request.form.get("conversion", "").strip()
    if not conversion:
        return None
    return float(conversion)


# --- Quick Summary ---
@app.route("/", methods=["GET", "POST"])
def quick_summary_page():
    if request.method == "GET":
        return render_template("quick_summary.html")

    generated_figures = []

    try:
        tracks_file, spots_file = get_uploaded_trackmate_files()
        conversion = parse_conversion_factor()

        # A temporary directory keeps uploaded files isolated and removes them
        # automatically after the report has been generated.
        with tempfile.TemporaryDirectory(dir=app.config["UPLOAD_FOLDER"]) as temp_dir:
            tracks_name = secure_filename(tracks_file.filename) or "tracks.csv"
            spots_name = secure_filename(spots_file.filename) or "spots.csv"
            tracks_path = os.path.join(temp_dir, f"tracks_{tracks_name}")
            spots_path = os.path.join(temp_dir, f"spots_{spots_name}")
            tracks_file.save(tracks_path)
            spots_file.save(spots_path)

            trajectory_fig = CT.trajectory(
                Tracks=tracks_path,
                Spots=spots_path,
                Conversion=conversion,
            )
            generated_figures.append(trajectory_fig)

            displacement_fig = CT.displacement(
                Tracks=tracks_path,
                Spots=spots_path,
                Conversion=conversion,
            )
            generated_figures.append(displacement_fig)

            speed_fig = CT.speed(
                Tracks=tracks_path,
                Spots=spots_path,
                Conversion=conversion,
            )
            generated_figures.append(speed_fig)

            # This axis-free image is the actual input used by the CNN.
            cnn_fig = CT.trajectory_CNN(
                Tracks=tracks_path,
                Spots=spots_path,
                Conversion=conversion,
            )
            generated_figures.append(cnn_fig)
            cnn_input_tensor = fig_to_tensor(cnn_fig)
            cnn_prediction, cnn_confidence = run_cnn_prediction(cnn_input_tensor)

            feature_df = CT.track_ML(
                Tracks=tracks_path,
                Spots=spots_path,
                Conversion=conversion,
            )

            missing_features = [
                feature for feature in FEATURE_ORDER if feature not in feature_df.columns
            ]
            if missing_features:
                missing_text = ", ".join(missing_features)
                raise ValueError(
                    f"The generated feature table is missing required columns: {missing_text}"
                )

            if feature_df.empty:
                raise ValueError("No XGBoost features were generated from the uploaded files.")

            input_data = feature_df.loc[:, FEATURE_ORDER].iloc[[0]].to_numpy(dtype=float)
            xgb_prediction, xgb_confidence = run_xgboost_prediction(input_data)

            display_features = build_display_features(feature_df.iloc[0])

            trajectory_url = fig_to_base64_img(trajectory_fig)
            displacement_url = fig_to_base64_img(displacement_fig)
            speed_url = fig_to_base64_img(speed_fig)

        models_agree = cnn_prediction == xgb_prediction
        agreement_text = (
            f"Both models predict {cnn_prediction}."
            if models_agree
            else "The CNN and XGBoost predictions do not agree."
        )

        return render_template(
            "quick_summary_result.html",
            tracks_filename=tracks_name,
            spots_filename=spots_name,
            trajectory_url=trajectory_url,
            displacement_url=displacement_url,
            speed_url=speed_url,
            cnn_prediction=cnn_prediction,
            cnn_confidence=cnn_confidence,
            xgb_prediction=xgb_prediction,
            xgb_confidence=xgb_confidence,
            models_agree=models_agree,
            agreement_text=agreement_text,
            input_features=display_features,
        )

    except ValueError as exc:
        return render_template("quick_summary.html", error=str(exc))
    except Exception as exc:
        return render_template(
            "quick_summary.html",
            error=f"An analysis error occurred: {exc}",
        )
    finally:
        for fig in generated_figures:
            try:
                plt.close(fig)
            except Exception:
                pass


# --- Individual CiliaTracks Analyses ---
@app.route("/analysis", methods=["GET", "POST"])
def analysis_page():
    if request.method == "POST":
        try:
            tracks_file, spots_file = get_uploaded_trackmate_files()
            tracks_path = os.path.join(
                app.config["UPLOAD_FOLDER"], secure_filename(tracks_file.filename)
            )
            spots_path = os.path.join(
                app.config["UPLOAD_FOLDER"], secure_filename(spots_file.filename)
            )
            tracks_file.save(tracks_path)
            spots_file.save(spots_path)
            conversion = parse_conversion_factor()
            action = request.form.get("action")
            fig, display_features, plot_title = None, None, ""

            if action == "trajectory":
                fig = CT.trajectory(Tracks=tracks_path, Spots=spots_path, Conversion=conversion)
                plot_title = "Trajectory Plot"
            elif action == "displacement":
                fig = CT.displacement(Tracks=tracks_path, Spots=spots_path, Conversion=conversion)
                plot_title = "Displacement Plot"
            elif action == "speed":
                fig = CT.speed(Tracks=tracks_path, Spots=spots_path, Conversion=conversion)
                plot_title = "Speed Plot"
            elif action == "cnn_input":
                fig = CT.trajectory_CNN(Tracks=tracks_path, Spots=spots_path, Conversion=conversion)
                plot_title = "CNN Input Image"
            elif action == "xgboost_features":
                df = CT.track_ML(Tracks=tracks_path, Spots=spots_path, Conversion=conversion)
                if df.empty:
                    raise ValueError("No XGBoost features were generated from the uploaded files.")
                missing_features = [
                    feature for feature in FEATURE_ORDER if feature not in df.columns
                ]
                if missing_features:
                    missing_text = ", ".join(missing_features)
                    raise ValueError(
                        f"The generated feature table is missing required columns: {missing_text}"
                    )
                display_features = build_display_features(df.iloc[0])
                plot_title = "XGBoost Input Features"
            else:
                raise ValueError("Please select an analysis action.")

            plot_url = fig_to_base64_img(fig) if fig else None
            if fig:
                plt.close(fig)

            return render_template(
                "analysis_result.html",
                plot_title=plot_title,
                plot_url=plot_url,
                input_features=display_features,
            )

        except Exception as exc:
            return render_template(
                "analysis_page.html",
                error=f"An error occurred: {exc}",
            )

    return render_template("analysis_page.html")


# --- CNN Model ---
@app.route("/cnn")
def cnn_page():
    return render_template("cnn_page.html")


@app.route("/predict_cnn", methods=["POST"])
def predict_cnn():
    form_type = request.form.get("form_type")
    generated_image_url = None
    image_filename = None

    if form_type == "automated":
        try:
            tracks_file, spots_file = get_uploaded_trackmate_files()
            tracks_path = os.path.join(
                app.config["UPLOAD_FOLDER"], secure_filename(tracks_file.filename)
            )
            spots_path = os.path.join(
                app.config["UPLOAD_FOLDER"], secure_filename(spots_file.filename)
            )
            tracks_file.save(tracks_path)
            spots_file.save(spots_path)
            conversion = parse_conversion_factor()

            cnn_fig = CT.trajectory_CNN(
                Tracks=tracks_path,
                Spots=spots_path,
                Conversion=conversion,
            )
            input_tensor = fig_to_tensor(cnn_fig)
            generated_image_url = fig_to_base64_img(cnn_fig)
            plt.close(cnn_fig)

        except Exception as exc:
            return render_template(
                "cnn_page.html",
                automated_error=f"An analysis error occurred: {exc}",
            )

    elif form_type == "manual":
        try:
            if "file" not in request.files or not request.files["file"].filename:
                return render_template(
                    "cnn_page.html",
                    manual_error="Please select an image file to upload.",
                )

            file = request.files["file"]
            image_filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], image_filename)
            file.save(filepath)
            input_tensor = preprocess_image(filepath)

        except Exception as exc:
            return render_template(
                "cnn_page.html",
                manual_error=f"An error occurred processing the image: {exc}",
            )
    else:
        return redirect(url_for("cnn_page"))

    prediction, confidence_score = run_cnn_prediction(input_tensor)
    return render_template(
        "result.html",
        model_type="CNN",
        prediction=prediction,
        confidence=confidence_score,
        image_url=generated_image_url,
        image_filename=image_filename,
    )


# --- XGBoost Model ---
@app.route("/xgboost", methods=["GET", "POST"])
def xgboost_page():
    if request.method == "POST":
        form_type = request.form.get("form_type")

        if form_type == "manual":
            try:
                features = [float(request.form[name]) for name in FEATURE_ORDER]
                feature_values = dict(zip(FEATURE_ORDER, features))
                display_features = build_display_features(feature_values)
                input_data = np.asarray(features, dtype=float).reshape(1, -1)
            except (ValueError, KeyError):
                return render_template(
                    "xgboost_page.html",
                    manual_error=(
                        "Invalid input. Please ensure all fields are filled with numbers."
                    ),
                )

        elif form_type == "automated":
            try:
                tracks_file, spots_file = get_uploaded_trackmate_files()
                tracks_path = os.path.join(
                    app.config["UPLOAD_FOLDER"], secure_filename(tracks_file.filename)
                )
                spots_path = os.path.join(
                    app.config["UPLOAD_FOLDER"], secure_filename(spots_file.filename)
                )
                tracks_file.save(tracks_path)
                spots_file.save(spots_path)
                conversion = parse_conversion_factor()

                feature_df = CT.track_ML(
                    Tracks=tracks_path,
                    Spots=spots_path,
                    Conversion=conversion,
                )
                input_data = feature_df.loc[:, FEATURE_ORDER].iloc[[0]].to_numpy(dtype=float)
                display_features = build_display_features(feature_df.iloc[0])

            except Exception as exc:
                return render_template(
                    "xgboost_page.html",
                    automated_error=f"An analysis error occurred: {exc}",
                )
        else:
            return redirect(url_for("xgboost_page"))

        prediction, confidence_score = run_xgboost_prediction(input_data)
        return render_template(
            "result.html",
            model_type="XGBoost",
            prediction=prediction,
            confidence=confidence_score,
            input_features=display_features,
        )

    return render_template("xgboost_page.html")


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


# --- Automatic browser opening and server start ---
def open_browser():
    """Wait briefly, then open the local application in the default browser."""
    webbrowser.open_new("http://127.0.0.1:5000/")


if __name__ == "__main__":
    Timer(1, open_browser).start()
    app.run(port=5000, debug=False)
