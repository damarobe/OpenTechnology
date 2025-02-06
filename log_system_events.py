import os
import json
import logging
from datetime import datetime

# Configuration
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "system_events.log")
EVENT_HISTORY_FILE = os.path.join(LOG_DIR, "event_history.json")

# Ensure log directory exists
os.makedirs(LOG_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def log_event(event_type, message, details=None):
    """
    Logs a system event with an optional details dictionary.
    
    :param event_type: Type of event (e.g., "INFO", "WARNING", "ERROR").
    :param message: Short message describing the event.
    :param details: Optional dictionary with additional event details.
    """
    event = {
        "timestamp": datetime.now().isoformat(),
        "event_type": event_type,
        "message": message,
        "details": details if details else {},
    }

    # Log event in standard log file
    log_message = f"{event_type} - {message} | Details: {json.dumps(details, indent=2) if details else 'None'}"
    
    if event_type == "ERROR":
        logging.error(log_message)
    elif event_type == "WARNING":
        logging.warning(log_message)
    else:
        logging.info(log_message)

    # Append event to JSON history file
    append_event_to_history(event)

    print(f"[{event_type}] {message}")


def append_event_to_history(event):
    """
    Appends an event to the event history JSON file.
    """
    if os.path.exists(EVENT_HISTORY_FILE):
        with open(EVENT_HISTORY_FILE, "r") as f:
            try:
                event_history = json.load(f)
            except json.JSONDecodeError:
                event_history = []
    else:
        event_history = []

    # Keep history to a reasonable limit (e.g., last 1000 events)
    event_history.append(event)
    event_history = event_history[-1000:]

    with open(EVENT_HISTORY_FILE, "w") as f:
        json.dump(event_history, f, indent=4)


def read_recent_events(limit=10):
    """
    Reads and prints the most recent system events from the event history file.
    
    :param limit: Number of recent events to display.
    """
    if not os.path.exists(EVENT_HISTORY_FILE):
        print("No event history found.")
        return

    with open(EVENT_HISTORY_FILE, "r") as f:
        try:
            event_history = json.load(f)
        except json.JSONDecodeError:
            print("Error reading event history.")
            return

    print(f"\n--- Last {min(limit, len(event_history))} Events ---")
    for event in event_history[-limit:]:
        print(f"{event['timestamp']} - {event['event_type']}: {event['message']}")
        if event["details"]:
            print(f"  Details: {json.dumps(event['details'], indent=2)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Logs system events and retrieves event history.")
    parser.add_argument("--read", action="store_true", help="Read the last 10 system events")
    parser.add_argument("--limit", type=int, default=10, help="Limit for the number of recent events to display")

    args = parser.parse_args()

    if args.read:
        read_recent_events(args.limit)
    else:
        log_event("INFO", "System event logging started", {"module": "log_system_events.py"})
