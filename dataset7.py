# See [Huggineface Transformer Tutorial](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)

import math
from typing import Tuple

import torch
from torch import Tensor
from torchtext.datasets import WikiText2
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import IterableDataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from config import PAD, SOS, EOS, UNK
from pathlib import Path


class Dataset7(Dataset):

    def __init__(self, raw_text_iter: IterableDataset, tokenizer: Tokenizer, bsz: int, bptt: int) -> None:
        super().__init__()

        # self.tokenizer: Tokenizer = tokenizer
        # self.vocab_size: int = tokenizer.get_vocab_size()
        # self.vocab = tokenizer.get_vocab()
        # self.raw_text_iter: IterableDataset = raw_text_iter
        self.processed_data: Tensor = self.data_process(raw_text_iter, tokenizer)
        self.batchified_data: Tensor = self.batchify(self.processed_data, bsz)
        self.bptt = bptt

    def data_process(self, raw_text_iter: IterableDataset, tokenizer: Tokenizer):
        """Converts raw text into a flat Tensor."""
        data = [torch.tensor(tokenizer.encode(item).ids, dtype=torch.long) for item in raw_text_iter]
        return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

    def batchify(self, processed_data: Tensor, bsz: int) -> Tensor:
        """Divides the data into ``bsz`` separate sequences, removing extra elements
        that wouldn't cleanly fit.

        Arguments:
            data: Tensor, shape ``[N]``
            bsz: int, batch size

        Returns:
            Tensor of shape ``[N // bsz, bsz]``
        """
        seq_len = processed_data.size(0) // bsz
        batchified_data = processed_data[:seq_len * bsz]
        batchified_data = batchified_data.view(bsz, seq_len).t().contiguous()
        # return data.to(device)
        return batchified_data

    def __len__(self):
        return len(self.batchified_data)

    def get_batch(self, i: int) -> Tuple[Tensor, Tensor]:
        """
        Args:
            source: Tensor, shape ``[full_seq_len, batch_size]``
            i: int

        Returns:
            tuple (data, target), where data has shape ``[seq_len, batch_size]`` and
            target has shape ``[seq_len * batch_size]``
        """
        seq_len = min(self.bptt, len(self.batchified_data) - 1 - i)
        data = self.batchified_data[i:i+seq_len]
        target = self.batchified_data[i+1:i+1+seq_len].reshape(-1)
        return data, target

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        # data = self.batchified_data[idx]
        # target = self.batchified_data[idx+1].reshape(-1)
        # return data, target
        return self.get_batch(idx)


def get_or_build_tokenizer7(config: dict, model_folder: str) -> Tokenizer:
    tokenizer_path = Path(model_folder + "/" + config['tokenizer_file'].format("en") + ".json")
    if not Path.exists(tokenizer_path):
        train_iter = WikiText2(split='train')
        tokenizer = Tokenizer(WordLevel(unk_token=UNK))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=[UNK, PAD, SOS, EOS], min_frequency=2)
        tokenizer.train_from_iterator(train_iter, trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_tokenizer7(config: dict, model_folder: str) -> Tokenizer:
    tokenizer_path = Path(model_folder + "/" + config['tokenizer_file'].format("en") + ".json")
    if not Path.exists(tokenizer_path):
        print(f"Tokenizer does not exists {tokenizer_path}")
        raise ValueError(f"{tokenizer_path} Tokenizer does not exist")
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_ds7(config: dict, model_folder: str) -> Tuple[DataLoader, DataLoader, DataLoader, Tokenizer]:

    tokenizer = get_or_build_tokenizer7(config, model_folder)

    # ``train_iter`` was "consumed" by the process of building the vocab,
    # so we have to create it again
    train_iter, val_iter, test_iter = WikiText2()
    train_data = Dataset7(train_iter, tokenizer=tokenizer, bsz=20, bptt=35)
    val_data = Dataset7(val_iter, tokenizer=tokenizer, bsz=10, bptt=35)
    test_data = Dataset7(test_iter, tokenizer=tokenizer, bsz=10, bptt=35)

    # JEB: Transformer does not want bs as first dimension.
    # JEB: This is a hack. Should probable able to use transpose instead
    # train_dataloader = DataLoader(train_data, config['batch_size'])
    train_dataloader = DataLoader(train_data, 1)
    val_dataloader = DataLoader(val_data, 1)
    test_dataloader = DataLoader(test_data, 1)

    return train_dataloader, val_dataloader, test_dataloader, tokenizer


def local_testing():
    vocab_iter = WikiText2(split='train')
    tokenizer = Tokenizer(WordLevel(unk_token=UNK))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(special_tokens=[UNK, PAD, SOS, EOS], min_frequency=2)
    tokenizer.train_from_iterator(vocab_iter, trainer=trainer)

    # ``train_iter`` was "consumed" by the process of building the vocab,
    # so we have to create it again
    bptt = 35
    train_iter = WikiText2(split='train')
    train_ds = Dataset7(train_iter, tokenizer, bsz=20, bptt=35)
    train_data = train_ds.batchified_data

    num_batches = len(train_data) // bptt
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = train_ds.get_batch(i)
        print('.................')
        print(data.shape)
        print(targets.shape)


if __name__ == "__main__":
    local_testing()
