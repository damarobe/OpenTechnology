import os
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt

# Configuration
FATIGUE_THRESHOLD_TYPING_SPEED = 2.0  # Keystrokes per second (KPS) below which fatigue is suspected
FATIGUE_THRESHOLD_PAUSE_VARIANCE = 0.1  # Variance in pause intervals above which fatigue is suspected
FATIGUE_THRESHOLD_AUDIO_VARIANCE = 50  # Spectral feature variance threshold for fatigue detection
ALERT_OUTPUT_FILE = "fatigue_alerts.json"


def load_feature_vectors(feature_vector_file):
    """
    Loads extracted feature vectors from a CSV file.
    """
    print(f"Loading feature vectors from: {feature_vector_file}")
    df = pd.read_csv(feature_vector_file)

    if df.empty:
        raise ValueError("Feature vector file is empty or corrupted.")

    return df.iloc[0].to_dict()


def detect_fatigue(features):
    """
    Detects fatigue based on predefined thresholds.
    """
    fatigue_flags = {}

    # Typing speed-based fatigue detection
    if features.get("mean_pause", 0) > FATIGUE_THRESHOLD_TYPING_SPEED:
        fatigue_flags["low_typing_speed"] = True

    # Variance in pause intervals (increased irregularity)
    if features.get("pause_variance", 0) > FATIGUE_THRESHOLD_PAUSE_VARIANCE:
        fatigue_flags["irregular_typing_pauses"] = True

    # Increased audio spectral irregularity
    if features.get("spectral_bandwidth_mean", 0) > FATIGUE_THRESHOLD_AUDIO_VARIANCE:
        fatigue_flags["high_audio_variability"] = True

    fatigue_detected = any(fatigue_flags.values())
    return fatigue_detected, fatigue_flags


def generate_fatigue_report(features, fatigue_detected, fatigue_flags, output_file):
    """
    Saves fatigue detection results and recommendations in a JSON file.
    """
    report = {
        "fatigue_detected": fatigue_detected,
        "alerts": fatigue_flags,
        "recommendations": []
    }

    # Provide recommendations based on detected fatigue patterns
    if "low_typing_speed" in fatigue_flags:
        report["recommendations"].append("Your typing speed is unusually low. Consider taking a short break.")

    if "irregular_typing_pauses" in fatigue_flags:
        report["recommendations"].append("Your typing rhythm is irregular. Try relaxing your hands and stretching.")

    if "high_audio_variability" in fatigue_flags:
        report["recommendations"].append("Your keystroke sounds vary significantly. Ensure a comfortable typing posture.")

    with open(output_file, "w") as f:
        json.dump(report, f, indent=4)

    print(f"Fatigue report saved: {output_file}")


def generate_fatigue_alerts(feature_vector_file, output_folder):
    """
    Full pipeline to analyze feature vectors, detect fatigue, and generate alerts.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Load extracted features
    features = load_feature_vectors(feature_vector_file)

    # Detect fatigue conditions
    fatigue_detected, fatigue_flags = detect_fatigue(features)

    # Save fatigue report
    output_file = os.path.join(output_folder, ALERT_OUTPUT_FILE)
    generate_fatigue_report(features, fatigue_detected, fatigue_flags, output_file)


if __name__ == "__main__":
    import argparse

    # Command-line argument parser
    parser = argparse.ArgumentParser(description="Analyzes typing and audio features to detect fatigue and generate alerts.")
    parser.add_argument("feature_vector_file", type=str, help="Path to the feature vector CSV file")
    parser.add_argument("output_folder", type=str, help="Folder to save fatigue detection results")

    args = parser.parse_args()

    # Run fatigue alert generation
    generate_fatigue_alerts(args.feature_vector_file, args.output_folder)
