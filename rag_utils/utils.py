import yaml
import os

def load_config(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Config file not found: {file_path}")

    with open(file_path, "r") as file:
        try:
            return yaml.safe_load(file)
        except yaml.YAMLError as e:
            raise RuntimeError(f"Error parsing YAML file: {e}")