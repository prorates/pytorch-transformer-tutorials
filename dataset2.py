import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Tuple
from datasets import load_dataset
from pathlib import Path


def get_ds2(config: dict, model_folder: str, device):
    # Generate random sample data
    # THis model2 does not care about language itself. Only generates random tokens
    fake_src_vocab_size = 500
    src_data = torch.randint(1, fake_src_vocab_size, (64, config['seq_len'])).to(device)  # (batch_size, seq_length)
    fake_tgt_vocab_size = 500
    tgt_data = torch.randint(1, fake_tgt_vocab_size, (64, config['seq_len'])).to(device)  # (batch_size, seq_length)

    # build tokenizers
    # tokenizer_src = get_or_build_tokenizer1(config, model_folder, ds_raw, config['lang_src'])
    # tokenizer_tgt = get_or_build_tokenizer1(config, model_folder, ds_raw, config['lang_tgt'])
    return src_data, tgt_data, fake_src_vocab_size, fake_tgt_vocab_size
