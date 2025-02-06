import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
PAUSE_THRESHOLD = 1.0  # Threshold (in seconds) for detecting long pauses (potential fatigue)
HISTOGRAM_BINS = 30  # Number of bins for histogram plotting
MOVING_AVERAGE_WINDOW = 10  # Window size for rolling average analysis


def load_keystroke_data(csv_file):
    """
    Loads keystroke event timestamps from a CSV file.
    """
    print(f"Loading keystroke data from: {csv_file}")
    df = pd.read_csv(csv_file)

    if "timestamp" not in df.columns:
        raise ValueError("CSV file must contain a 'timestamp' column.")

    return df["timestamp"].values


def compute_pause_intervals(keystrokes):
    """
    Computes pause intervals (time differences) between consecutive keystrokes.
    """
    if len(keystrokes) < 2:
        print("Not enough data to compute pause intervals.")
        return None

    pause_intervals = np.diff(keystrokes)  # Compute time differences

    # Compute statistical metrics
    pause_stats = {
        "mean_pause": np.mean(pause_intervals),
        "median_pause": np.median(pause_intervals),
        "max_pause": np.max(pause_intervals),
        "min_pause": np.min(pause_intervals),
        "std_pause": np.std(pause_intervals),
        "long_pauses_count": np.sum(pause_intervals > PAUSE_THRESHOLD),
    }

    print("\nPause Interval Statistics:")
    for key, value in pause_stats.items():
        print(f"{key}: {value:.3f} sec")

    return pause_intervals, pause_stats


def plot_pause_intervals(pause_intervals, output_file):
    """
    Generates a histogram and time-series plot for pause intervals.
    """
    plt.figure(figsize=(12, 5))
    
    # Histogram
    plt.subplot(1, 2, 1)
    sns.histplot(pause_intervals, bins=HISTOGRAM_BINS, kde=True, color="blue")
    plt.xlabel("Pause Duration (s)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Pause Intervals")

    # Time-series plot
    plt.subplot(1, 2, 2)
    plt.plot(range(len(pause_intervals)), pause_intervals, marker="o", linestyle="-", color="red", alpha=0.6)
    plt.xlabel("Keystroke Event Index")
    plt.ylabel("Pause Duration (s)")
    plt.title("Pause Intervals Over Time")

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"Pause interval plots saved: {output_file}")


def save_pause_data(keystrokes, pause_intervals, output_file):
    """
    Saves computed pause intervals to a CSV file.
    """
    data = {
        "timestamp": keystrokes[1:],  # Skip first keystroke since diff() reduces count
        "pause_intervals": pause_intervals,
    }
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Pause intervals saved: {output_file}")


def analyze_pause_intervals(csv_file, output_folder):
    """
    Full pipeline to compute and analyze pause intervals from keystroke timestamps.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Load keystroke data
    keystrokes = load_keystroke_data(csv_file)

    # Compute pause intervals
    pause_intervals, pause_stats = compute_pause_intervals(keystrokes)

    if pause_intervals is None:
        print("Insufficient keystroke data for analysis.")
        return

    # Save results
    base_filename = os.path.splitext(os.path.basename(csv_file))[0]
    csv_output = os.path.join(output_folder, f"{base_filename}_pause_intervals.csv")
    plot_output = os.path.join(output_folder, f"{base_filename}_pause_analysis.png")

    save_pause_data(keystrokes, pause_intervals, csv_output)
    plot_pause_intervals(pause_intervals, plot_output)


if __name__ == "__main__":
    import argparse

    # Command-line argument parser
    parser = argparse.ArgumentParser(description="Analyzes pause intervals between keystrokes.")
    parser.add_argument("csv_file", type=str, help="Path to the keystroke event CSV file")
    parser.add_argument("output_folder", type=str, help="Folder to save pause interval analysis")

    args = parser.parse_args()

    # Run pause interval analysis
    analyze_pause_intervals(args.csv_file, args.output_folder)
