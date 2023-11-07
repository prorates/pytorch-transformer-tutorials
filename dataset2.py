import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, random_split
from torch.autograd import Variable
from typing import Any
import numpy as np

from typing import Tuple

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path
from config import EOS, SOS, PAD, UNK


def get_ds2_old(config: dict, model_folder: str, device) -> Tuple[Tensor, Tensor, int, int]:
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


class Dataset2(Dataset):

    def __init__(self, ds, t_src: Tokenizer, t_tgt: Tokenizer, src_lang: str, tgt_lang: str, seq_len: int) -> None:
        super().__init__()

        self.ds = ds
        self.t_src = t_src
        self.t_tgt = t_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        self.sos_token = torch.tensor([t_tgt.token_to_id(SOS)], dtype=torch.int64)
        self.eos_token = torch.tensor([t_tgt.token_to_id(EOS)], dtype=torch.int64)
        self.pad_token = torch.tensor([t_tgt.token_to_id(PAD)], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: Any) -> Any:
        src_target_pair = self.ds[idx]
        src_text = src_target_pair[self.src_lang]
        tgt_text = src_target_pair[self.tgt_lang]

        # Transform the text into tokens
        enc_input_tokens = self.t_src.encode(src_text).ids
        dec_input_tokens = self.t_tgt.encode(tgt_text).ids

        # need to pad the sentence to sequence lenght.
        # minus two because we add start and end
        # Add sos, eos and padding to each sentence
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2

        # minus one because we add start and end
        # We will only add <s>, and </s> only on the label
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        # Make sure the number of padding tokens is not negative. If it is, the sentence is too long
        if (enc_num_padding_tokens < 0) or (dec_num_padding_tokens < 0):
            raise ValueError('Sentence is too long')

        # Add SOS and EOS to source text
        # Model1 and Model3 are different. Nothing seems to be added to SRC
        # SRC = data.Field(lower=True, tokenize=t_src.tokenizer)
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Add SOS to decoder input
        # Model1 and Model3 are different. Both SOS and EOS are added
        # tgt = data.Field(lower=True, tokenize=t_tgt.tokenizer, init_token=SOS, eos_token=EOS)
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Add EOS to label
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Double check the size of the tensors to make sure they are all seq_len long
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        src_mask, tgt_mask = self.generate_mask(encoder_input, decoder_input)

        return {
            "src": encoder_input,
            "tgt": decoder_input,
            "src_mask": src_mask,
            "tgt_mask": tgt_mask,
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text
        }

    def generate_mask(self, src, tgt):
        # src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        # tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        src_mask = (src != self.pad_token).unsqueeze(-2)
        tgt_mask = (tgt != self.pad_token).unsqueeze(-2)
        seq_length = tgt.size(0)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        # One more time issue with the nopeak_mask
        # tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask


def get_all_sentences2(ds, lang):
    for item in ds:
        yield item[lang]


# JEB: Original code was based on depraced torchtext.data.Field amd spacy
# Migrating to Vocab and get_tokenizer did not seem to be worth it.
# Using the HuggingFace Tokenizer instead
def get_or_build_tokenizer2(config: dict, model_folder: str, ds, lang: str) -> Tokenizer:
    tokenizer_path = Path(model_folder + "/" + config['tokenizer_file'].format(lang) + ".json")
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token=UNK))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=[UNK, PAD, SOS, EOS], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences2(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_tokenizer2(config: dict, model_folder: str, lang: str) -> Tokenizer:
    tokenizer_path = Path(model_folder + "/" + config['tokenizer_file'].format(lang) + ".json")
    if not Path.exists(tokenizer_path):
        print(f"Tokenizer does not exists {tokenizer_path}")
        raise ValueError(f"{tokenizer_path} Tokenizer does not exist")
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_ds2(config: dict, model_folder: str) -> Tuple[DataLoader, DataLoader, Tokenizer, Tokenizer]:
    # load_dataset(path, name, split=)
    ds_raw = load_dataset(
        "csv", data_files=f"custom_datasets/{config['datasource']}_{config['lang_src']}_{config['lang_tgt']}/dataset.csv", sep="|", split='train')

    # build tokenizers
    tokenizer_src = get_or_build_tokenizer2(config, model_folder, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer2(config, model_folder, ds_raw, config['lang_tgt'])

    # keep 90% for training and 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = Dataset2(train_ds_raw, tokenizer_src, tokenizer_tgt,
                        config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = Dataset2(val_ds_raw, tokenizer_src, tokenizer_tgt,
                      config['lang_src'], config['lang_tgt'], config['seq_len'])

    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item[config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item[config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_testing_ds2(config: dict, model_folder: str, sentence: str) -> Tuple[str, str, Tokenizer, Tokenizer]:

    # build tokenizers
    tokenizer_src = get_tokenizer2(config, model_folder, config['lang_src'])
    tokenizer_tgt = get_tokenizer2(config, model_folder, config['lang_tgt'])

    # keep 90% for training and 10% for validation
    label = ""
    if isinstance(sentence, int) or sentence.isdigit():
        id = int(sentence)
        ds = load_dataset(
            "csv",
            data_files=f"custom_datasets/{config['datasource']}_{config['lang_src']}_{config['lang_tgt']}/dataset.csv",
            sep="|",
            split='all')
        ds = Dataset2(ds, tokenizer_src, tokenizer_tgt,
                      config['lang_src'], config['lang_tgt'], config['seq_len'])
        sentence = ds[id]['src_text']
        label = ds[id]["tgt_text"]

    return sentence, label, tokenizer_src, tokenizer_tgt
