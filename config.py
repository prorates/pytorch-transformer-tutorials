from typing import Optional
from pathlib import Path
import os
import torch
import yaml

EOS = "<eos>"
SOS = "<sos>"
UNK = "<unk>"
PAD = "<pad>"


def get_default_config() -> dict:
    return {
        "batch_size": 8,
        "block_size": 32,  # Added for model2
        "num_epochs": 20,
        "lr": 10**-4,
        "seq_len": 80,
        "d_model": 256,  # paper: 512,
        "N": 6,  # paper: 6,
        "h": 8,  # paper: 8,
        "dropout": 10**-1,
        "d_ff": 1024,  # paper 2048, # supposed to be "4*d_model"
        # "datasource": 'opus_books',
        # "datasource": 'translate',
        "datasource": "tinyshakespeare",
        "lang_src": "en",
        "lang_tgt": "en",
        "model_basename": "tmodel_",
        "preload": "latest",  # Possible values: None, "02", "latest"
        "tokenizer_file": "tokenizer_{0}",
        "experiment_name": "runs/tmodel",
        "alt_model": "model8",  # Possible values: None, model1, model2
    }


def get_config(filename: Optional[str] = None, modelfolder: Optional[str] = None) -> dict:
    default_config = get_default_config()
    if modelfolder:
        config_path = Path(modelfolder + "/" + "config.yaml")
    elif filename:
        config_path = Path(filename)
    else:
        modelfolder = get_model_folder(default_config)
        Path(modelfolder).mkdir(parents=True, exist_ok=True)
        config_path = Path(modelfolder + "/" + "config.yaml")
        if not Path.exists(config_path):
            with open(config_path, "w") as write:
                yaml.dump(default_config, write)

    if not Path.exists(config_path):
        print("Using default config from config.py")
        return default_config
    else:
        print(f"Loading config from {config_path}")
        with open(config_path, "r") as yamlFile:
            configdict = yaml.safe_load(yamlFile)
            return configdict


def get_device():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_built() or torch.backends.mps.is_available() else "cpu"
    if device == "cuda":
        print(f"Using NVIDIA GPU and device {device}")
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
    elif device == "mps":
        print(f"Using Apple Silicon and device {device}")
    else:
        print(f"Using device {device}")
    return device


def get_console_width():
    try:
        # get the console window width
        with os.popen("stty size", "r") as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except BaseException:
        # If we can't get the console width, use 80 as default
        console_width = 80
    return console_width


def get_model_folder(config: dict):
    if config["alt_model"]:
        return f"{config['datasource']}_{config['lang_src']}_{config['lang_tgt']}_{config['alt_model']}"
    else:
        return f"{config['datasource']}_{config['lang_src']}_{config['lang_tgt']}"


def get_weights_file_path(config: dict, epoch: str):
    model_folder = get_model_folder(config)
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path(".") / model_folder / model_filename)


def get_best_model_params_path(config: dict):
    model_folder = get_model_folder(config)
    model_filename = "best_model_params.pt"
    return str(Path(".") / model_folder / model_filename)


def latest_weights_file_path(config: dict):
    model_folder = get_model_folder(config)
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])
