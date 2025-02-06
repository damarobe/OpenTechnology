import os
import numpy as np
import librosa
import librosa.display
import scipy.signal
import noisereduce as nr
from pydub import AudioSegment

# Configuration Parameters
AUDIO_FORMAT = "m4a"  # Target format
SAMPLE_RATE = 44100  # Standard sample rate for audio processing
TARGET_DURATION = 10  # Target length of each segment in seconds
NOISE_REDUCTION = True  # Enable noise reduction
SILENCE_TRIM_THRESHOLD = -40  # dB threshold for silence trimming
NORMALIZE_VOLUME = True  # Enable volume normalization


def load_audio(input_file):
    """
    Load an audio file into a NumPy array with a fixed sample rate.
    """
    audio, sr = librosa.load(input_file, sr=SAMPLE_RATE, mono=True)
    return audio, sr


def normalize_audio(audio):
    """
    Normalize the volume of the audio signal.
    """
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        return audio / max_val  # Normalize between -1 and 1
    return audio


def reduce_noise(audio, sr):
    """
    Apply noise reduction to the audio signal.
    """
    return nr.reduce_noise(y=audio, sr=sr)


def trim_silence(audio):
    """
    Trim leading and trailing silence from the audio signal.
    """
    trimmed_audio, _ = librosa.effects.trim(audio, top_db=SILENCE_TRIM_THRESHOLD)
    return trimmed_audio


def segment_audio(audio, sr):
    """
    Split the audio into equal-length segments.
    """
    segment_length = sr * TARGET_DURATION  # Number of samples per segment
    num_segments = len(audio) // segment_length
    return np.array_split(audio, num_segments)


def save_audio(audio, sr, output_file):
    """
    Save the processed audio as an M4A file using pydub.
    """
    # Convert NumPy array to a pydub AudioSegment
    audio_segment = AudioSegment(
        (audio * 32767).astype(np.int16).tobytes(),
        frame_rate=sr,
        sample_width=2,
        channels=1,
    )
    audio_segment.export(output_file, format=AUDIO_FORMAT)
    print(f"Processed audio saved: {output_file}")


def preprocess_audio(input_file, output_folder):
    """
    Full pipeline to preprocess audio: load, normalize, trim silence, reduce noise, segment, and save.
    """
    print(f"Processing {input_file}...")

    # Step 1: Load audio
    audio, sr = load_audio(input_file)

    # Step 2: Normalize volume
    if NORMALIZE_VOLUME:
        audio = normalize_audio(audio)

    # Step 3: Trim silence
    audio = trim_silence(audio)

    # Step 4: Noise reduction
    if NOISE_REDUCTION:
        audio = reduce_noise(audio, sr)

    # Step 5: Split into segments
    audio_segments = segment_audio(audio, sr)

    # Step 6: Save processed segments
    base_filename = os.path.splitext(os.path.basename(input_file))[0]
    for i, segment in enumerate(audio_segments):
        output_file = os.path.join(output_folder, f"{base_filename}_segment_{i}.{AUDIO_FORMAT}")
        save_audio(segment, sr, output_file)


if __name__ == "__main__":
    import argparse

    # Command-line argument parser
    parser = argparse.ArgumentParser(description="Preprocesses audio files for typing fatigue detection.")
    parser.add_argument("input_file", type=str, help="Path to the input audio file (.m4a)")
    parser.add_argument("output_folder", type=str, help="Path to the folder where processed audio will be saved")

    args = parser.parse_args()

    # Ensure output folder exists
    os.makedirs(args.output_folder, exist_ok=True)

    # Run preprocessing
    preprocess_audio(args.input_file, args.output_folder)
