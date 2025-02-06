import os
import numpy as np
import librosa
import librosa.display
import scipy.signal
import matplotlib.pyplot as plt
from pydub import AudioSegment

# Configuration Parameters
SAMPLE_RATE = 44100  # Standard sampling rate
MIN_PEAK_HEIGHT = 0.1  # Normalized threshold for keystroke detection
MIN_PEAK_DISTANCE = 0.05  # Minimum time between keystrokes (seconds)

def load_audio(input_file):
    """
    Loads an audio file and converts it to a NumPy array.
    """
    print(f"Loading audio file: {input_file}")
    audio, sr = librosa.load(input_file, sr=SAMPLE_RATE, mono=True)
    return audio, sr

def detect_keystroke_events(audio, sr):
    """
    Detects keystroke events by identifying peaks in the waveform.
    """
    print("Detecting keystroke events...")

    # Compute envelope using Short-Time Energy (STE)
    frame_length = int(sr * 0.01)  # 10ms window
    hop_length = frame_length // 2
    energy = np.array([
        sum(abs(audio[i : i + frame_length] ** 2))
        for i in range(0, len(audio), hop_length)
    ])

    # Normalize energy
    energy = energy / np.max(energy)

    # Find peaks in energy
    peak_indices, _ = scipy.signal.find_peaks(
        energy, height=MIN_PEAK_HEIGHT, distance=int(MIN_PEAK_DISTANCE * sr / hop_length)
    )

    # Convert peak indices to timestamps
    keystroke_times = peak_indices * hop_length / sr

    print(f"Detected {len(keystroke_times)} keystrokes.")
    return keystroke_times, energy, peak_indices, hop_length

def plot_waveform(audio, sr, keystroke_times, output_file):
    """
    Plots the waveform with detected keystrokes.
    """
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(audio, sr=sr, alpha=0.6)
    
    # Mark detected keystrokes
    for t in keystroke_times:
        plt.axvline(x=t, color="r", linestyle="--", alpha=0.6)

    plt.title("Keystroke Detection")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.savefig(output_file)
    plt.close()
    print(f"Waveform plot saved: {output_file}")

def save_keystroke_events(keystroke_times, output_file):
    """
    Saves the extracted keystroke timestamps to a CSV file.
    """
    np.savetxt(output_file, keystroke_times, delimiter=",", header="timestamp", comments="")
    print(f"Keystroke events saved: {output_file}")

def extract_keystroke_events(input_file, output_folder):
    """
    Full pipeline: load audio, detect keystrokes, save results.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Load audio
    audio, sr = load_audio(input_file)

    # Detect keystrokes
    keystroke_times, energy, peak_indices, hop_length = detect_keystroke_events(audio, sr)

    # Save detected keystrokes
    base_filename = os.path.splitext(os.path.basename(input_file))[0]
    csv_output = os.path.join(output_folder, f"{base_filename}_keystrokes.csv")
    plot_output = os.path.join(output_folder, f"{base_filename}_waveform.png")

    save_keystroke_events(keystroke_times, csv_output)
    plot_waveform(audio, sr, keystroke_times, plot_output)

if __name__ == "__main__":
    import argparse

    # Command-line argument parser
    parser = argparse.ArgumentParser(description="Extracts keystroke events from an audio file.")
    parser.add_argument("input_file", type=str, help="Path to the input audio file (.m4a)")
    parser.add_argument("output_folder", type=str, help="Folder to save extracted keystroke events")

    args = parser.parse_args()

    # Run extraction
    extract_keystroke_events(args.input_file, args.output_folder)
