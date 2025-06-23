"""
config.py

This module is responsible for loading the simulation parameters from the
config.yaml file and making them available as a global Python object.
This allows any other module in the project to easily access configuration
settings.
"""

import yaml

def load_config(config_path='config.yaml'):
    """
    Loads the YAML configuration file.

    Args:
        config_path (str): The path to the config.yaml file.

    Returns:
        dict: A dictionary containing the configuration parameters.
    """
    try:
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        print("Configuration loaded successfully from config.yaml.")
        return config_data
    except FileNotFoundError:
        print(f"Error: Configuration file not found at '{config_path}'.")
        print("Please ensure config.yaml is in the project's root directory.")
        return None
    except Exception as e:
        print(f"Error loading or parsing config.yaml: {e}")
        return None

# Load the configuration once when the module is imported.
# This creates a global CONFIG object accessible to other scripts.
CONFIG = load_config()

