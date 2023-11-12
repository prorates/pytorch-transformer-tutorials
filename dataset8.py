import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Tuple
from torch.utils.data import DataLoader
from tokenizers import Tokenizer


def get_batch(split, train_data, val_data, block_size: int, batch_size: int, device):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


def get_ds8(config: dict, model_folder: str) -> Tuple[DataLoader, DataLoader, DataLoader, Tokenizer]:
    return None, None, None, None


def local_testing():
    # wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    # here are all the unique characters that occur in this text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    # create a mapping from characters to integers
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    def encode(s): return [stoi[c] for c in s]  # encoder: take a string, output a list of integers
    def decode(l): return ''.join([itos[i] for i in l])  # decoder: take a list of integers, output a string

    # Train and test splits
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9*len(data))  # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]

# data loading


if __name__ == "__main__":
    local_testing()