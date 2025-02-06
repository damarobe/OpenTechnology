import os
import time
import argparse
import shutil
from datetime import datetime, timedelta

# Configuration
DEFAULT_FOLDERS = {
    "session_logs": "session_logs",  # Folder storing session data
    "features": "output_folder",  # Folder containing feature vectors
    "audio_files": "audio_recordings"  # Folder storing raw and processed audio files
}
FILE_RETENTION_DAYS = 7  # Number of days to keep files before deleting
LOG_FILE = "cleanup_log.txt"  # Log file to record cleanup actions


def is_old_file(file_path, days=FILE_RETENTION_DAYS):
    """
    Checks if a file is older than the specified retention period.
    """
    file_age_days = (time.time() - os.path.getmtime(file_path)) / 86400
    return file_age_days > days


def delete_old_files(folder, extensions=None, days=FILE_RETENTION_DAYS):
    """
    Deletes files in a folder that are older than the specified retention period.
    If extensions are provided, only deletes files with matching extensions.
    """
    if not os.path.exists(folder):
        print(f"Skipping cleanup: {folder} does not exist.")
        return

    deleted_files = []
    for root, _, files in os.walk(folder):
        for file in files:
            file_path = os.path.join(root, file)
            if extensions and not file.endswith(extensions):
                continue  # Skip files with non-matching extensions
            if is_old_file(file_path, days):
                os.remove(file_path)
                deleted_files.append(file_path)

    log_cleanup(folder, deleted_files)


def delete_empty_folders(folder):
    """
    Deletes empty subdirectories within a folder.
    """
    if not os.path.exists(folder):
        return

    for root, dirs, _ in os.walk(folder, topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            if not os.listdir(dir_path):  # Check if directory is empty
                os.rmdir(dir_path)
                print(f"Deleted empty folder: {dir_path}")


def log_cleanup(folder, deleted_files):
    """
    Logs cleanup activity into a file.
    """
    if not deleted_files:
        return

    with open(LOG_FILE, "a") as log:
        log.write(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Cleanup performed in: {folder}\n")
        for file in deleted_files:
            log.write(f"Deleted: {file}\n")

    print(f"Cleanup completed in {folder}. {len(deleted_files)} files removed.")


def cleanup_system(retention_days=FILE_RETENTION_DAYS):
    """
    Runs the cleanup process for session logs, feature vectors, and audio files.
    """
    print(f"Starting data cleanup (Retention Period: {retention_days} days)...")

    # Delete old session logs
    delete_old_files(DEFAULT_FOLDERS["session_logs"], extensions=(".json",), days=retention_days)

    # Delete old feature vectors
    delete_old_files(DEFAULT_FOLDERS["features"], extensions=(".csv",), days=retention_days)

    # Delete old audio recordings
    delete_old_files(DEFAULT_FOLDERS["audio_files"], extensions=(".m4a", ".wav"), days=retention_days)

    # Remove empty folders
    for folder in DEFAULT_FOLDERS.values():
        delete_empty_folders(folder)

    print("Data cleanup completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deletes old session logs, feature vectors, and audio files.")
    parser.add_argument("--days", type=int, default=FILE_RETENTION_DAYS, help="Number of days to retain data before deletion")
    args = parser.parse_args()

    # Run cleanup
    cleanup_system(args.days)
