from typing import Tuple

import torch
from torch import Tensor
from torchtext.datasets import WikiText2
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import IterableDataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from config import PAD, SOS, EOS, UNK
from pathlib import Path


class Dataset7(Dataset):

    def __init__(self, raw_text_iter: IterableDataset, tokenizer_tgt: Tokenizer, bptt: int) -> None:
        super().__init__()
        self.tokenizer_tgt = tokenizer_tgt
        self.bptt = bptt
        self.raw_text_iter = raw_text_iter

    def data_process(self) -> Tensor:
        """Converts raw text into a flat Tensor."""
        data = [torch.tensor(vocab(self.tokenizer_tgt(item)), dtype=torch.long) for item in self.raw_text_iter]
        return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

    def batchify(self, data: Tensor, bsz: int) -> Tensor:
        """Divides the data into ``bsz`` separate sequences, removing extra elements
        that wouldn't cleanly fit.

        Arguments:
            data: Tensor, shape ``[N]``
            bsz: int, batch size

        Returns:
            Tensor of shape ``[N // bsz, bsz]``
        """
        seq_len = data.size(0) // bsz
        data = data[:seq_len * bsz]
        data = data.view(bsz, seq_len).t().contiguous()
        # return data.to(device)
        return data

    def get_batch(self, source: Tensor, i: int) -> Tuple[Tensor, Tensor]:
        """
        Args:
            source: Tensor, shape ``[full_seq_len, batch_size]``
            i: int

        Returns:
            tuple (data, target), where data has shape ``[seq_len, batch_size]`` and
            target has shape ``[seq_len * batch_size]``
        """
        seq_len = min(self.bptt, len(source) - 1 - i)
        data = source[i:i+seq_len]
        target = source[i+1:i+1+seq_len].reshape(-1)
        return data, target


def get_or_build_tokenizer7(config: dict, model_folder: str) -> Tokenizer:
    tokenizer_path = Path(model_folder + "/" + config['tokenizer_file'].format("en") + ".json")
    if not Path.exists(tokenizer_path):
        train_iter = WikiText2(split='train')
        # tokenizer = get_tokenizer('basic_english')
        # vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=[UNK])
        # vocab.set_default_index(vocab[UNK])
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
    train_data = Dataset7(train_iter, tokenizer_tgt=tokenizer, bptt=35)
    val_data = Dataset7(val_iter, tokenizer_tgt=tokenizer, bptt=35)
    test_data = Dataset7(test_iter, tokenizer_tgt=tokenizer, bptt=35)

    train_dataloader = DataLoader(train_data, config['batch_size'])
    val_dataloader = DataLoader(val_data, 1)
    test_dataloader = DataLoader(test_data, 1)

    return train_dataloader, val_dataloader, test_dataloader, tokenizer
