import pyaudio
import wave
import numpy as np
from pydub import AudioSegment
import time

# Recording parameters
FORMAT = pyaudio.paInt16  # 16-bit audio format
CHANNELS = 1  # Mono recording
RATE = 44100  # Sample rate in Hz
CHUNK = 1024  # Buffer size
DURATION = 10  # Recording duration in seconds
OUTPUT_WAV = "typing_audio.wav"
OUTPUT_M4A = "typing_audio.m4a"

def record_audio(duration, output_file):
    """
    Records audio using the computer's microphone and saves it as a WAV file.
    """
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("Recording... Press any key to stop.")

    frames = []
    start_time = time.time()

    try:
        while time.time() - start_time < duration:
            data = stream.read(CHUNK)
            frames.append(data)

    except KeyboardInterrupt:
        print("\nRecording stopped.")

    print("Recording complete. Saving file...")

    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save recorded audio as WAV file
    with wave.open(output_file, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    print(f"Audio saved as {output_file}")

def convert_to_m4a(wav_file, m4a_file):
    """
    Converts a WAV file to M4A format using pydub.
    """
    audio = AudioSegment.from_wav(wav_file)
    audio.export(m4a_file, format="mp4")  # M4A uses mp4 container
    print(f"Converted to {m4a_file}")

if __name__ == "__main__":
    # Record audio and save as WAV
    record_audio(DURATION, OUTPUT_WAV)

    # Convert WAV to M4A
    convert_to_m4a(OUTPUT_WAV, OUTPUT_M4A)
