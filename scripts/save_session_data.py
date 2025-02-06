import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

# Configuration
SESSION_LOG_DIR = "session_logs"  # Default directory for session logs
SESSION_FILE_TEMPLATE = "session_{timestamp}.json"  # Session file naming pattern


def load_feature_vectors(feature_vector_file):
    """
    Loads extracted feature vectors from a CSV file.
    """
    print(f"Loading feature vectors from: {feature_vector_file}")
    df = pd.read_csv(feature_vector_file)

    if df.empty:
        raise ValueError("Feature vector file is empty or corrupted.")

    return df.iloc[0].to_dict()


def load_fatigue_alerts(fatigue_alerts_file):
    """
    Loads fatigue detection results from a JSON file.
    """
    if os.path.exists(fatigue_alerts_file):
        with open(fatigue_alerts_file, "r") as f:
            return json.load(f)
    return {"fatigue_detected": False, "alerts": {}, "recommendations": []}


def save_session_data(feature_vector_file, fatigue_alerts_file, output_folder):
    """
    Saves the typing session data (features + fatigue alerts) into a structured JSON file.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Load extracted features and fatigue detection results
    features = load_feature_vectors(feature_vector_file)
    fatigue_data = load_fatigue_alerts(fatigue_alerts_file)

    # Get session timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session_file = os.path.join(output_folder, SESSION_FILE_TEMPLATE.format(timestamp=timestamp))

    # Combine session data
    session_data = {
        "session_id": timestamp,
        "timestamp": datetime.now().isoformat(),
        "typing_features": features,
        "fatigue_analysis": fatigue_data,
    }

    # Save session data as JSON
    with open(session_file, "w") as f:
        json.dump(session_data, f, indent=4)

    print(f"Session data saved: {session_file}")


if __name__ == "__main__":
    import argparse

    # Command-line argument parser
    parser = argparse.ArgumentParser(description="Saves typing session data including feature vectors and fatigue analysis.")
    parser.add_argument("feature_vector_file", type=str, help="Path to the feature vector CSV file")
    parser.add_argument("fatigue_alerts_file", type=str, help="Path to the fatigue analysis JSON file")
    parser.add_argument("output_folder", type=str, help="Folder to save session data")

    args = parser.parse_args()

    # Run session data saving
    save_session_data(args.feature_vector_file, args.fatigue_alerts_file, args.output_folder)
