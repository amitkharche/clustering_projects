"""
Utility functions for logging and common tasks.
"""
import logging
import os

def setup_logging(log_file="logs/training.log"):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
