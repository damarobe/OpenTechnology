import os
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt

# Configuration
FEATURES_DIR = "features"  # Default directory to look for extracted features
OUTPUT_FILE = "feature_vectors.csv"


def load_keystroke_features(keystroke_file):
    """
    Loads keystroke timing features from a CSV file.
    """
    print(f"Loading keystroke features from: {keystroke_file}")
    df = pd.read_csv(keystroke_file)
    
    # Ensure required columns exist
    if "pause_intervals" not in df.columns:
        raise ValueError("Keystroke feature file must contain 'pause_intervals' column.")

    # Compute additional statistical features
    pause_intervals = df["pause_intervals"].dropna()
    features = {
        "mean_pause": np.mean(pause_intervals),
        "median_pause": np.median(pause_intervals),
        "max_pause": np.max(pause_intervals),
        "min_pause": np.min(pause_intervals),
        "std_pause": np.std(pause_intervals),
        "pause_variance": np.var(pause_intervals),
        "long_pause_ratio": np.sum(pause_intervals > 1.0) / len(pause_intervals),
    }

    return features


def load_audio_features(audio_feature_file):
    """
    Loads audio spectral and MFCC features from a CSV file.
    """
    print(f"Loading audio features from: {audio_feature_file}")
    df = pd.read_csv(audio_feature_file)

    # Convert the row to a dictionary
    features = df.iloc[0].to_dict()
    
    return features


def merge_features(keystroke_features, audio_features):
    """
    Combines keystroke timing and audio spectral features into a single feature vector.
    """
    feature_vector = {**keystroke_features, **audio_features}
    return feature_vector


def save_feature_vectors(feature_vectors, output_file):
    """
    Saves aggregated feature vectors to a CSV file.
    """
    df = pd.DataFrame(feature_vectors)
    df.to_csv(output_file, index=False)
    print(f"Feature vectors saved: {output_file}")


def generate_feature_vectors(keystroke_file, audio_feature_file, output_folder):
    """
    Full pipeline to generate structured feature vectors for machine learning.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Load extracted features
    keystroke_features = load_keystroke_features(keystroke_file)
    audio_features = load_audio_features(audio_feature_file)

    # Merge features
    feature_vector = merge_features(keystroke_features, audio_features)

    # Save the feature vector
    output_file = os.path.join(output_folder, OUTPUT_FILE)
    save_feature_vectors([feature_vector], output_file)


if __name__ == "__main__":
    import argparse

    # Command-line argument parser
    parser = argparse.ArgumentParser(description="Generates feature vectors for typing fatigue analysis.")
    parser.add_argument("keystroke_file", type=str, help="Path to the keystroke timing features CSV file")
    parser.add_argument("audio_feature_file", type=str, help="Path to the audio features CSV file")
    parser.add_argument("output_folder", type=str, help="Folder to save the feature vectors")

    args = parser.parse_args()

    # Run feature vector generation
    generate_feature_vectors(args.keystroke_file, args.audio_feature_file, args.output_folder)
