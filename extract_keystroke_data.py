import os
import numpy as np
import librosa
import librosa.display
import scipy.signal
import matplotlib.pyplot as plt
from pydub import AudioSegment

# Configuration
SAMPLE_RATE = 44100  # Standard sampling rate
MIN_PEAK_HEIGHT = 0.05  # Normalized threshold for keystroke detection
MIN_PEAK_DISTANCE = 0.05  # Minimum time (seconds) between keystrokes
NOISE_REDUCTION = True  # Apply noise reduction

def load_audio(input_file):
    """
    Loads an audio file and converts it to a NumPy array.
    """
    print(f"Loading audio file: {input_file}")
    audio, sr = librosa.load(input_file, sr=SAMPLE_RATE, mono=True)
    return audio, sr

def spectral_analysis(audio, sr):
    """
    Computes the spectral energy to enhance keystroke detection.
    """
    print("Performing spectral analysis...")

    # Compute Short-Time Fourier Transform (STFT)
    stft = np.abs(librosa.stft(audio, n_fft=1024, hop_length=512))

    # Compute spectral energy (sum of squared magnitudes)
    spectral_energy = np.sum(stft**2, axis=0)

    # Normalize spectral energy
    spectral_energy = spectral_energy / np.max(spectral_energy)

    return spectral_energy

def detect_keystrokes(audio, sr):
    """
    Detects keystroke events based on spectral energy analysis.
    """
    print("Detecting keystrokes...")

    # Compute spectral energy
    spectral_energy = spectral_analysis(audio, sr)

    # Find peaks in spectral energy
    peak_indices, _ = scipy.signal.find_peaks(
        spectral_energy, height=MIN_PEAK_HEIGHT, distance=int(MIN_PEAK_DISTANCE * sr / 512)
    )

    # Convert peak indices to timestamps
    keystroke_times = peak_indices * 512 / sr

    print(f"Detected {len(keystroke_times)} keystrokes.")
    return keystroke_times, spectral_energy, peak_indices

def plot_spectral_energy(spectral_energy, keystroke_times, output_file):
    """
    Plots the spectral energy and detected keystroke events.
    """
    plt.figure(figsize=(12, 4))
    plt.plot(spectral_energy, color="b", label="Spectral Energy")
    
    for t in keystroke_times:
        plt.axvline(x=t, color="r", linestyle="--", alpha=0.6)

    plt.title("Keystroke Detection using Spectral Energy")
    plt.xlabel("Time (s)")
    plt.ylabel("Energy")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_file)
    plt.close()
    print(f"Spectral energy plot saved: {output_file}")

def save_keystroke_data(keystroke_times, output_file):
    """
    Saves the extracted keystroke timestamps to a CSV file.
    """
    np.savetxt(output_file, keystroke_times, delimiter=",", header="timestamp", comments="")
    print(f"Keystroke timestamps saved: {output_file}")

def extract_keystroke_data(input_file, output_folder):
    """
    Full pipeline to extract keystroke timestamps from an audio file.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Load audio
    audio, sr = load_audio(input_file)

    # Detect keystrokes
    keystroke_times, spectral_energy, peak_indices = detect_keystrokes(audio, sr)

    # Save results
    base_filename = os.path.splitext(os.path.basename(input_file))[0]
    csv_output = os.path.join(output_folder, f"{base_filename}_keystrokes.csv")
    plot_output = os.path.join(output_folder, f"{base_filename}_spectral_energy.png")

    save_keystroke_data(keystroke_times, csv_output)
    plot_spectral_energy(spectral_energy, keystroke_times, plot_output)

if __name__ == "__main__":
    import argparse

    # Command-line argument parser
    parser = argparse.ArgumentParser(description="Extracts keystroke data from an audio file using spectral analysis.")
    parser.add_argument("input_file", type=str, help="Path to the input audio file (.m4a)")
    parser.add_argument("output_folder", type=str, help="Folder to save extracted keystroke data")

    args = parser.parse_args()

    # Run extraction
    extract_keystroke_data(args.input_file, args.output_folder)
