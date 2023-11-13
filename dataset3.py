# The source code seems to be [here](https://github.com/SamLynnEvans/Transformer?ref=blog.floydhub.com)

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


def nopeak_mask(size: int) -> Tensor:
    np_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    np_mask = Variable(torch.from_numpy(np_mask == 0))
    return np_mask


class Dataset3(Dataset):

    def __init__(self, ds, t_src: Tokenizer, t_trg: Tokenizer, src_lang: str, tgt_lang: str, seq_len: int) -> None:
        super().__init__()

        self.ds = ds
        self.t_src = t_src
        self.t_trg = t_trg
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        self.sos_token = torch.tensor([t_trg.token_to_id(SOS)], dtype=torch.int64)
        self.eos_token = torch.tensor([t_trg.token_to_id(EOS)], dtype=torch.int64)
        self.pad_token = torch.tensor([t_trg.token_to_id(PAD)], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: Any) -> Any:
        src_target_pair = self.ds[idx]
        src_text = src_target_pair[self.src_lang]
        tgt_text = src_target_pair[self.tgt_lang]

        # Transform the text into tokens
        enc_input_tokens = self.t_src.encode(src_text).ids
        dec_input_tokens = self.t_trg.encode(tgt_text).ids

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
        # TRG = data.Field(lower=True, tokenize=t_trg.tokenizer, init_token='<sos>', eos_token='<eos>')
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

        return {
            "src": encoder_input,
            "trg": decoder_input,
            "src_mask": (encoder_input != self.pad_token).unsqueeze(-2),
            # JEB: Issue here & nopeak_mask(decoder_input.size(0)),
            "trg_mask": (decoder_input != self.pad_token).unsqueeze(-2),
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text
        }

    def nopeak_mask(self, size):
        np_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
        np_mask = Variable(torch.from_numpy(np_mask == 0))
        return np_mask

    def create_masks(self, src, trg):
        src_mask = (src != self.pad_token).unsqueeze(-2)

        if trg is not None:
            trg_mask = (trg != self.pad_token).unsqueeze(-2)
            size = trg.size(1)  # get seq_len for matrix
            np_mask = nopeak_mask(size)
            trg_mask = trg_mask & np_mask
        else:
            trg_mask = None
        return src_mask, trg_mask

    # def create_fields(self, opt):
    #     TRG = data.Field(lower=True, tokenize=t_trg.tokenizer, init_token='<sos>', eos_token='<eos>')
    #     SRC = data.Field(lower=True, tokenize=t_src.tokenizer)
    #     return (SRC, TRG)

    # def create_dataset(self, opt, SRC, TRG):
    #     opt.src_pad = SRC.vocab.stoi['<pad>']
    #     opt.trg_pad = TRG.vocab.stoi['<pad>']


def get_all_sentences3(ds, lang):
    for item in ds:
        yield item[lang]


# Migrating to Vocab and get_tokenizer did not seem to be worth it.
# Using the HuggingFace Tokenizer instead
def get_or_build_tokenizer3(config: dict, model_folder: str, ds, lang: str) -> Tokenizer:
    tokenizer_path = Path(model_folder + "/" + config['tokenizer_file'].format(lang) + ".json")
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token=UNK))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=[UNK, PAD, SOS, EOS], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences3(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_tokenizer3(config: dict, model_folder: str, lang: str) -> Tokenizer:
    tokenizer_path = Path(model_folder + "/" + config['tokenizer_file'].format(lang) + ".json")
    if not Path.exists(tokenizer_path):
        print(f"Tokenizer does not exists {tokenizer_path}")
        raise ValueError(f"{tokenizer_path} Tokenizer does not exist")
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_ds3(config: dict, model_folder: str) -> Tuple[DataLoader, DataLoader, Tokenizer, Tokenizer]:
    # load_dataset(path, name, split=)
    ds_raw = load_dataset(
        "csv", data_files=f"custom_datasets/{config['datasource']}_{config['lang_src']}_{config['lang_tgt']}/dataset.csv", sep="|", split='train')

    # build tokenizers
    tokenizer_src = get_or_build_tokenizer3(config, model_folder, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer3(config, model_folder, ds_raw, config['lang_tgt'])

    # keep 90% for training and 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = Dataset3(train_ds_raw, tokenizer_src, tokenizer_tgt,
                        config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = Dataset3(val_ds_raw, tokenizer_src, tokenizer_tgt,
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


def get_testing_ds3(config: dict, model_folder: str, sentence: str) -> Tuple[str, str, Tokenizer, Tokenizer]:

    # build tokenizers
    tokenizer_src = get_tokenizer3(config, model_folder, config['lang_src'])
    tokenizer_tgt = get_tokenizer3(config, model_folder, config['lang_tgt'])

    # keep 90% for training and 10% for validation
    label = ""
    if isinstance(sentence, int) or sentence.isdigit():
        id = int(sentence)
        ds = load_dataset(
            "csv",
            data_files=f"custom_datasets/{config['datasource']}_{config['lang_src']}_{config['lang_tgt']}/dataset.csv",
            sep="|",
            split='all')
        ds = Dataset3(ds, tokenizer_src, tokenizer_tgt,
                      config['lang_src'], config['lang_tgt'], config['seq_len'])
        sentence = ds[id]['src_text']
        label = ds[id]["tgt_text"]

    return sentence, label, tokenizer_src, tokenizer_tgt
