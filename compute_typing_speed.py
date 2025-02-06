import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Configuration
TYPING_SPEED_WINDOW = 10  # Window size in seconds for rolling KPS calculation


def load_keystroke_data(csv_file):
    """
    Loads keystroke event timestamps from a CSV file.
    """
    print(f"Loading keystroke data from: {csv_file}")
    df = pd.read_csv(csv_file)
    
    if "timestamp" not in df.columns:
        raise ValueError("CSV file must contain a 'timestamp' column.")
    
    return df["timestamp"].values


def compute_typing_speed(keystrokes):
    """
    Computes typing speed (keystrokes per second) and pause statistics.
    """
    if len(keystrokes) < 2:
        print("Not enough data to compute typing speed.")
        return None, None

    # Compute time differences (pause intervals) between keystrokes
    pause_intervals = np.diff(keystrokes)

    # Compute typing speed (KPS)
    typing_speed = 1 / pause_intervals  # Keystrokes per second

    # Compute rolling KPS over a time window
    rolling_kps = []
    for i in range(len(keystrokes)):
        start_time = keystrokes[i]
        end_time = start_time + TYPING_SPEED_WINDOW
        num_keystrokes = np.sum((keystrokes >= start_time) & (keystrokes < end_time))
        rolling_kps.append(num_keystrokes / TYPING_SPEED_WINDOW)

    # Compute statistical metrics
    pause_stats = {
        "mean_pause": np.mean(pause_intervals),
        "median_pause": np.median(pause_intervals),
        "max_pause": np.max(pause_intervals),
        "min_pause": np.min(pause_intervals),
        "std_pause": np.std(pause_intervals),
    }

    return typing_speed, pause_intervals, rolling_kps, pause_stats


def plot_typing_speed(keystrokes, rolling_kps, output_file):
    """
    Generates a plot showing typing speed over time.
    """
    time_axis = keystrokes[: len(rolling_kps)]  # Match lengths

    plt.figure(figsize=(10, 4))
    plt.plot(time_axis, rolling_kps, marker="o", linestyle="-", color="b", label="Typing Speed (KPS)")
    plt.xlabel("Time (s)")
    plt.ylabel("Keystrokes Per Second (KPS)")
    plt.title("Typing Speed Over Time")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_file)
    plt.close()
    print(f"Typing speed plot saved: {output_file}")


def save_typing_speed_data(keystrokes, typing_speed, pause_intervals, output_file):
    """
    Saves computed typing speed and pause intervals as a CSV file.
    """
    data = {
        "timestamp": keystrokes[1:],
        "typing_speed": typing_speed,
        "pause_intervals": pause_intervals,
    }
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Typing speed data saved: {output_file}")


def compute_and_save_typing_speed(csv_file, output_folder):
    """
    Full pipeline to compute typing speed from keystroke timestamps.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Load keystroke data
    keystrokes = load_keystroke_data(csv_file)

    # Compute typing speed
    typing_speed, pause_intervals, rolling_kps, pause_stats = compute_typing_speed(keystrokes)

    if typing_speed is None:
        print("Insufficient keystroke data for meaningful analysis.")
        return

    # Save results
    base_filename = os.path.splitext(os.path.basename(csv_file))[0]
    csv_output = os.path.join(output_folder, f"{base_filename}_typing_speed.csv")
    plot_output = os.path.join(output_folder, f"{base_filename}_typing_speed_plot.png")

    save_typing_speed_data(keystrokes, typing_speed, pause_intervals, csv_output)
    plot_typing_speed(keystrokes, rolling_kps, plot_output)

    # Display pause statistics
    print("\nPause Statistics:")
    for key, value in pause_stats.items():
        print(f"{key}: {value:.3f} sec")


if __name__ == "__main__":
    import argparse

    # Command-line argument parser
    parser = argparse.ArgumentParser(description="Computes typing speed from keystroke event timestamps.")
    parser.add_argument("csv_file", type=str, help="Path to the keystroke event CSV file")
    parser.add_argument("output_folder", type=str, help="Folder to save computed typing speed data")

    args = parser.parse_args()

    # Run typing speed computation
    compute_and_save_typing_speed(args.csv_file, args.output_folder)
