import os
import numpy as np
import librosa
import librosa.display
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats

# Configuration
SAMPLE_RATE = 44100  # Audio sample rate
FRAME_LENGTH = 2048  # Frame size for spectral analysis
HOP_LENGTH = 512  # Hop size for feature extraction
MFCC_FEATURES = 13  # Number of MFCC coefficients to extract


def load_audio(input_file):
    """
    Loads an audio file and converts it to a NumPy array.
    """
    print(f"Loading audio file: {input_file}")
    audio, sr = librosa.load(input_file, sr=SAMPLE_RATE, mono=True)
    return audio, sr


def extract_audio_features(audio, sr):
    """
    Extracts key acoustic features from the audio signal.
    Returns a dictionary of computed features.
    """
    print("Extracting audio features...")

    # Compute spectral features
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
    zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)[0]

    # Compute MFCC (Mel-Frequency Cepstral Coefficients)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=MFCC_FEATURES)

    # Aggregate statistics (mean, std, max, min)
    features = {
        "spectral_centroid_mean": np.mean(spectral_centroid),
        "spectral_bandwidth_mean": np.mean(spectral_bandwidth),
        "spectral_rolloff_mean": np.mean(spectral_rolloff),
        "zero_crossing_rate_mean": np.mean(zero_crossing_rate),
        "mfcc_mean": [np.mean(mfcc) for mfcc in mfccs],
        "mfcc_std": [np.std(mfcc) for mfcc in mfccs],
    }

    return features


def plot_audio_features(audio, sr, output_file):
    """
    Plots the waveform and key spectral features.
    """
    plt.figure(figsize=(12, 6))

    # Plot waveform
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(audio, sr=sr, alpha=0.6)
    plt.title("Audio Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    # Plot spectral centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    times = librosa.times_like(spectral_centroid)
    plt.subplot(2, 1, 2)
    plt.plot(times, spectral_centroid, color="red", label="Spectral Centroid")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"Feature plots saved: {output_file}")


def save_audio_features(features, output_file):
    """
    Saves extracted audio features to a CSV file.
    """
    # Convert MFCCs to named columns
    feature_data = {key: [value] if isinstance(value, float) else value for key, value in features.items()}
    
    df = pd.DataFrame(feature_data)
    df.to_csv(output_file, index=False)
    print(f"Audio features saved: {output_file}")


def compute_audio_features(input_file, output_folder):
    """
    Full pipeline to compute audio features from keystroke recordings.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Load audio
    audio, sr = load_audio(input_file)

    # Extract features
    features = extract_audio_features(audio, sr)

    # Save results
    base_filename = os.path.splitext(os.path.basename(input_file))[0]
    csv_output = os.path.join(output_folder, f"{base_filename}_audio_features.csv")
    plot_output = os.path.join(output_folder, f"{base_filename}_waveform.png")

    save_audio_features(features, csv_output)
    plot_audio_features(audio, sr, plot_output)


if __name__ == "__main__":
    import argparse

    # Command-line argument parser
    parser = argparse.ArgumentParser(description="Extracts acoustic features from keystroke audio.")
    parser.add_argument("input_file", type=str, help="Path to the input audio file (.m4a)")
    parser.add_argument("output_folder", type=str, help="Folder to save extracted features")

    args = parser.parse_args()

    # Run audio feature extraction
    compute_audio_features(args.input_file, args.output_folder)
