# See [video](https://youtu.be/kCc8FmEb1nY)
# The colab repo is [here](https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing)


import torch
from torch import Tensor
from typing import Tuple
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path
from config import EOS, SOS, PAD, UNK, get_config, get_model_folder


class Dataset8(Dataset):

    def __init__(self, raw_text: str, tokenizer: Tokenizer, batch_size: int, block_size: int) -> None:
        super().__init__()

        self.processed_data: Tensor = self.data_process(raw_text, tokenizer)
        self.block_size = block_size
        self.batch_size = batch_size

    def data_process(self, raw_text: str, tokenizer: Tokenizer):
        """Converts raw text into a flat Tensor."""
        data = torch.tensor(tokenizer.encode(raw_text).ids, dtype=torch.long)
        return data

    def __len__(self):
        return len(self.processed_data) // self.block_size

    def get_batch(self) -> Tuple[Tensor, Tensor]:
        # generate a small batch of data of inputs x and targets y
        ix = torch.randint(len(self.processed_data) - self.block_size, (self.batch_size,))
        x = torch.stack([self.processed_data[i:i+self.block_size] for i in ix])
        y = torch.stack([self.processed_data[i+1:i+self.block_size+1] for i in ix])
        return x, y

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        # The iterator is supposed to stack them up to batch_size
        x = self.processed_data[idx:idx+self.block_size]
        y = self.processed_data[idx+1:idx+self.block_size+1]
        return x, y


def get_or_build_tokenizer8(config: dict, model_folder: str, ds: str, lang: str) -> Tokenizer:
    tokenizer_path = Path(model_folder + "/" + config['tokenizer_file'].format(lang) + ".json")
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(BPE(char_level=True, unk_token=UNK))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(special_tokens=[UNK, PAD, SOS, EOS, ' ', '?', '!'], max_token_length=1, min_frequency=1)
        tokenizer.train_from_iterator(sorted(list(set(ds))), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_tokenizer8(config: dict, model_folder: str, lang: str) -> Tokenizer:
    tokenizer_path = Path(model_folder + "/" + config['tokenizer_file'].format(lang) + ".json")
    if not Path.exists(tokenizer_path):
        print(f"Tokenizer does not exists {tokenizer_path}")
        raise ValueError(f"{tokenizer_path} Tokenizer does not exist")
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def load_custom_dataset(config: dict, model_folder: str) -> str:
    src_file = f"custom_datasets/{config['datasource']}/{config['lang_src']}.txt"

    with open(src_file, 'r', encoding='utf-8') as file:
        # src_sentences = file.readlines()
        src_sentences = file.read()

    return src_sentences


def get_ds8(config: dict, model_folder: str) -> Tuple[DataLoader, DataLoader, Tokenizer]:

    raw_text = load_custom_dataset(config, model_folder)
    tokenizer = get_or_build_tokenizer8(config, model_folder, raw_text, config['lang_src'])

    # keep 90% for training and 10% for validation
    # train_ds_size = int(0.9 * len(ds_raw))
    # val_ds_size = len(ds_raw) - train_ds_size
    # train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    n = int(0.9*len(raw_text))  # first 90% will be train, rest val

    batch_size = config['batch_size']
    block_size = 32  # what is the maximum context length for predictions?
    train_ds = Dataset8(raw_text[:n], tokenizer=tokenizer, batch_size=batch_size, block_size=block_size)
    val_ds = Dataset8(raw_text[n:], tokenizer=tokenizer, batch_size=batch_size, block_size=block_size)

    train_dataloader = DataLoader(train_ds, shuffle=True, batch_size=batch_size)
    val_dataloader = DataLoader(val_ds, 1)

    return train_dataloader, val_dataloader, tokenizer

def get_testing_ds8(config: dict, model_folder: str) -> Tokenizer:

    # build tokenizers
    tokenizer = get_tokenizer8(config, model_folder, config['lang_src'])

    return tokenizer

def local_testing():

    config = get_config()
    modelfolder = get_model_folder(config)
    raw_text = load_custom_dataset(config, modelfolder)

    tokenizer = get_or_build_tokenizer8(config, modelfolder, raw_text, "en")

    # data = torch.tensor(tokenizer.encode(raw_text).ids, dtype=torch.long)
    n = int(0.9*len(raw_text))  # first 90% will be train, rest val

    torch.manual_seed(1337)
    batch_size = 4
    block_size = 8
    train_ds = Dataset8(raw_text[:n], tokenizer=tokenizer, batch_size=batch_size, block_size=block_size)
    val_ds = Dataset8(raw_text[n:], tokenizer=tokenizer, batch_size=batch_size, block_size=block_size)
    train_dataloader = DataLoader(train_ds, shuffle=True, batch_size=batch_size)
    for batch_num, batch_iterator in enumerate(train_dataloader):
        xb, yb = batch_iterator
        print("inputs:")
        print(xb.shape)
        print(xb)
        print("target:")
        print(yb.shape)
        print(yb)

        for b in range(batch_size):  # batch dimension
            for t in range(block_size):  # time dimension
                context = xb[b, :t+1]
                target = yb[b, t]
                print(f"when input is {context.tolist()} the target: {target}")

        if (batch_num == 0):
            break


if __name__ == "__main__":
    local_testing()
