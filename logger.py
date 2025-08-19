# logger.py
import time
import sys

def log(message, level=None):  # Add optional level parameter
    """Maintains existing behavior but silently ignores level parameter"""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {message}", file=sys.stderr)
    sys.stderr.flush()