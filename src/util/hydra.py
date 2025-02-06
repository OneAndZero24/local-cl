import os
from pathlib import Path

from omegaconf import DictConfig


def extract_output_dir(config: DictConfig) -> Path:
    """
    Extracts path to output directory created by Hydra as pathlib.Path instance
    """
    
    date = '/'.join(list(config._metadata.resolver_cache['now'].values()))
    output_dir = Path.cwd() / os.environ['OUTPUT_DIR'] / date
    return output_dir

def preprocess_config(config: DictConfig):
    """
    Sets config.exp.log_dir to date extracted from metadata.
    """

    config.exp.log_dir = extract_output_dir(config)