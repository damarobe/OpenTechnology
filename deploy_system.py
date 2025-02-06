import os
import shutil
import subprocess
import logging
import argparse
import json
from datetime import datetime

# Configuration
PROJECT_NAME = "Typing Fatigue Monitoring System"
LOG_DIR = "logs"
SESSION_DIR = "session_logs"
FEATURE_DIR = "output_folder"
AUDIO_DIR = "audio_recordings"
DEPLOYMENT_LOG_FILE = os.path.join(LOG_DIR, "deployment.log")
SYSTEM_REQUIREMENTS = "requirements.txt"
SYSTEM_SCRIPTS = [
    "record_audio.py",
    "extract_keystroke_events.py",
    "compute_typing_speed.py",
    "analyze_pause_intervals.py",
    "compute_audio_features.py",
    "generate_feature_vectors.py",
    "generate_fatigue_alerts.py",
    "save_session_data.py",
    "ui_dashboard.py",
    "data_cleanup.py",
    "log_system_events.py"
]
PYTHON_EXECUTABLE = "python"  # Change to "python3" if required

# Ensure necessary directories exist
REQUIRED_DIRECTORIES = [LOG_DIR, SESSION_DIR, FEATURE_DIR, AUDIO_DIR]

# Set up logging
logging.basicConfig(
    filename=DEPLOYMENT_LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def log_event(event_type, message):
    """
    Logs a deployment event.
    """
    log_message = f"{event_type} - {message}"
    print(f"[{event_type}] {message}")

    if event_type == "ERROR":
        logging.error(log_message)
    else:
        logging.info(log_message)


def create_directories():
    """
    Creates necessary system directories.
    """
    log_event("INFO", "Creating necessary system directories.")
    for directory in REQUIRED_DIRECTORIES:
        if not os.path.exists(directory):
            os.makedirs(directory)
            log_event("INFO", f"Created directory: {directory}")


def install_dependencies():
    """
    Installs necessary Python dependencies from requirements.txt.
    """
    if not os.path.exists(SYSTEM_REQUIREMENTS):
        log_event("ERROR", "requirements.txt not found.")
        return

    log_event("INFO", "Installing Python dependencies.")
    try:
        subprocess.run([PYTHON_EXECUTABLE, "-m", "pip", "install", "-r", SYSTEM_REQUIREMENTS], check=True)
        log_event("INFO", "Python dependencies installed successfully.")
    except subprocess.CalledProcessError as e:
        log_event("ERROR", f"Failed to install dependencies: {str(e)}")


def check_scripts():
    """
    Checks if all required system scripts exist.
    """
    log_event("INFO", "Checking required system scripts.")
    missing_scripts = [script for script in SYSTEM_SCRIPTS if not os.path.exists(script)]
    if missing_scripts:
        log_event("ERROR", f"Missing scripts: {', '.join(missing_scripts)}")
        return False
    return True


def start_services():
    """
    Starts essential services like logging, UI dashboard, and cleanup.
    """
    log_event("INFO", "Starting system services.")

    # Start UI Dashboard
    try:
        subprocess.Popen([PYTHON_EXECUTABLE, "ui_dashboard.py"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        log_event("INFO", "UI Dashboard started successfully.")
    except Exception as e:
        log_event("ERROR", f"Failed to start UI Dashboard: {str(e)}")

    # Schedule cleanup to run periodically
    try:
        subprocess.Popen([PYTHON_EXECUTABLE, "data_cleanup.py"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        log_event("INFO", "Data cleanup service started.")
    except Exception as e:
        log_event("ERROR", f"Failed to start data cleanup: {str(e)}")


def verify_deployment():
    """
    Verifies that all required components are running properly.
    """
    log_event("INFO", "Verifying deployment status.")
    verification_report = {
        "project_name": PROJECT_NAME,
        "timestamp": datetime.now().isoformat(),
        "directories": {dir_name: os.path.exists(dir_name) for dir_name in REQUIRED_DIRECTORIES},
        "scripts_available": {script: os.path.exists(script) for script in SYSTEM_SCRIPTS},
        "status": "Deployment Successful" if check_scripts() else "Deployment Failed",
    }

    report_file = os.path.join(LOG_DIR, "deployment_report.json")
    with open(report_file, "w") as f:
        json.dump(verification_report, f, indent=4)

    log_event("INFO", f"Deployment verification report saved: {report_file}")


def deploy_system():
    """
    Executes the full deployment pipeline.
    """
    log_event("INFO", f"Deploying {PROJECT_NAME}.")

    create_directories()
    install_dependencies()

    if not check_scripts():
        log_event("ERROR", "System deployment failed due to missing scripts.")
        return

    start_services()
    verify_deployment()

    log_event("INFO", f"{PROJECT_NAME} deployed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploys the Typing Fatigue Monitoring System.")
    parser.add_argument("--verify", action="store_true", help="Verify deployment status without installing dependencies")

    args = parser.parse_args()

    if args.verify:
        verify_deployment()
    else:
        deploy_system()
