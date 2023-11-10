from pathlib import Path
import os
import torch

EOS = '<eos>'
SOS = '<sos>'
UNK = '<unk>'
PAD = '<pad>'


def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 20,
        "lr": 10**-4,
        "seq_len": 80,
        "d_model": 256,  # paper: 512,
        "N": 3,  # paper: 6,
        "h": 4,  # paper: 8,
        "dropout": 10**-1,
        "d_ff": 1024,  # paper 2048,
        # "datasource": 'opus_books',
        "datasource": 'translate',
        "lang_src": "en",
        "lang_tgt": "fr",
        "model_basename": "tmodel_",
        "preload": "latest",  # Possible values: None, "02", "latest"
        "tokenizer_file": "tokenizer_{0}",
        "experiment_name": "runs/tmodel",
        "alt_model": "model6"  # Possible values: None, model1, model2
    }


def get_device():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_built(
    ) or torch.backends.mps.is_available() else "cpu"
    if (device == 'cuda'):
        print(f'Using NVIDIA GPU and device {device}')
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
    elif (device == 'mps'):
        print(f'Using Apple Silicon and device {device}')
    else:
        print(f'Using device {device}')
    return device


def get_console_width():
    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except BaseException:
        # If we can't get the console width, use 80 as default
        console_width = 80
    return console_width


def get_model_folder(config):
    if config['alt_model']:
        return f"{config['datasource']}_{config['lang_src']}_{config['lang_tgt']}_{config['alt_model']}"
    else:
        return f"{config['datasource']}_{config['lang_src']}_{config['lang_tgt']}"


def get_weights_file_path(config, epoch: str):
    model_folder = get_model_folder(config)
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)


def latest_weights_file_path(config):
    model_folder = get_model_folder(config)
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])
