import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, random_split
from torch.autograd import Variable
from typing import Any

from typing import Tuple

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path


class Dataset1(Dataset):

    def __init__(self, ds, tokenizer_src: Tokenizer, tokenizer_tgt: Tokenizer,
                 src_lang: str, tgt_lang: str, seq_len: int) -> None:
        super().__init__()

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        # JEB: Big difference between video and source code.
        # DIFF1: tensor instead of Tensor
        # DIFF2: tgt instead of src
        # self.sos_token = torch.Tensor([tokenizer_src.token_to_id(['[SOS]'])], dtype=torch.int64)
        # self.eos_token = torch.Tensor([tokenizer_src.token_to_id(['[EOS]'])], dtype=torch.int64)
        # self.pad_token = torch.Tensor([tokenizer_src.token_to_id(['[PAD]'])], dtype=torch.int64)
        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: Any) -> Any:
        src_target_pair = self.ds[idx]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        # Transform the text into tokens
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

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
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            # (1, 1, Seq_Len) # padding mask should not be used in self attention
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            # (1, SeqLen) & (1, SeqLen, SeqLen)
            # JEB: Be carreful. Have to unsqueeze
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & casual_mask(decoder_input.size(0)),
            "label": label,  # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text
        }


def casual_mask(size: int) -> Tensor:
    # Everything above the diagonal needs to become 0
    # JEB: To check. Syntax is different in provided file and video
    # mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0


def translation_mask(size: int) -> Tensor:
    # Everything above the diagonal needs to become 0
    # JEB: Translate was not applying the == 0
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask


def get_all_sentences1(ds, lang):
    for item in ds:
        yield item['translation'][lang]


def get_or_build_tokenizer1(config: dict, model_folder: str, ds, lang: str) -> Tokenizer:
    tokenizer_path = Path(model_folder + "/" + config['tokenizer_file'].format(lang) + ".json")
    if not Path.exists(tokenizer_path):
        # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences1(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_ds1(config: dict, model_folder: str) -> Tuple[DataLoader, DataLoader, Tokenizer, Tokenizer]:
    # load_dataset(path, name, split=)
    ds_raw = load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split='train')

    # build tokenizers
    tokenizer_src = get_or_build_tokenizer1(config, model_folder, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer1(config, model_folder, ds_raw, config['lang_tgt'])

    # keep 90% for training and 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = Dataset1(train_ds_raw, tokenizer_src, tokenizer_tgt,
                        config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = Dataset1(val_ds_raw, tokenizer_src, tokenizer_tgt,
                      config['lang_src'], config['lang_tgt'], config['seq_len'])

    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt
