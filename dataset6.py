import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, random_split
from torch.autograd import Variable
from typing import Any

from typing import Tuple

from datasets import load_dataset
from tokenizers import Tokenizer, pre_tokenizers, decoders
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel, Whitespace

from pathlib import Path
from config import EOS, SOS, PAD, UNK
import numpy as np
import codecs

# Generated this by filtering Appendix code

START_TOKEN = SOS
PADDING_TOKEN = PAD
END_TOKEN = EOS

kannada_vocabulary = [START_TOKEN, ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/',
                      '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', '<', '=', '>', '?', 'ˌ',
                      'ँ', 'ఆ', 'ఇ', 'ా', 'ి', 'ీ', 'ు', 'ూ',
                      'ಅ', 'ಆ', 'ಇ', 'ಈ', 'ಉ', 'ಊ', 'ಋ', 'ೠ', 'ಌ', 'ಎ', 'ಏ', 'ಐ', 'ಒ', 'ಓ', 'ಔ',
                      'ಕ', 'ಖ', 'ಗ', 'ಘ', 'ಙ',
                      'ಚ', 'ಛ', 'ಜ', 'ಝ', 'ಞ',
                      'ಟ', 'ಠ', 'ಡ', 'ಢ', 'ಣ',
                      'ತ', 'ಥ', 'ದ', 'ಧ', 'ನ',
                      'ಪ', 'ಫ', 'ಬ', 'ಭ', 'ಮ',
                      'ಯ', 'ರ', 'ಱ', 'ಲ', 'ಳ', 'ವ', 'ಶ', 'ಷ', 'ಸ', 'ಹ',
                      '಼', 'ಽ', 'ಾ', 'ಿ', 'ೀ', 'ು', 'ೂ', 'ೃ', 'ೄ', 'ೆ', 'ೇ', 'ೈ', 'ೊ', 'ೋ', 'ೌ', '್', 'ೕ', 'ೖ', 'ೞ', 'ೣ', 'ಂ', 'ಃ',
                      '೦', '೧', '೨', '೩', '೪', '೫', '೬', '೭', '೮', '೯', PADDING_TOKEN, END_TOKEN]

english_vocabulary = [START_TOKEN, ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/',
                      '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                      ':', '<', '=', '>', '?', '@',
                      '[', '\\', ']', '^', '_', '`',
                      'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
                      'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
                      'y', 'z',
                      '{', '|', '}', '~', PADDING_TOKEN, END_TOKEN]

NEG_INFTY = -1e9


class Dataset6(Dataset):

    def __init__(self, src_sentences: list[str], tgt_sentences: list[str]):
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences

    def __len__(self) -> int:
        return len(self.src_sentences)

    def __getitem__(self, idx: int) -> Tuple[str, str]:
        return self.src_sentences[idx], self.tgt_sentences[idx]

    @staticmethod
    def create_masks(eng_batch: tuple[str], kn_batch: tuple[str], seq_len: int) -> Tuple[Tensor, Tensor, Tensor]:
        batch_size = len(eng_batch)
        # Create a tensor (SeqLen, SeqLen). Cell above the diagonals are set to True, cell bellow to 0.
        look_ahead_mask = torch.full([seq_len, seq_len], True)  # (SeqLen, SeqLen)
        look_ahead_mask = torch.triu(look_ahead_mask, diagonal=1)  # (SeqLen, SeqLen)
        # Create three tensors (bs, SeqLen, SeqLen)
        # (bs, SeqLen, SeqLen) filled with False
        encoder_padding_mask = torch.full([batch_size, seq_len, seq_len], False)
        decoder_padding_mask_self_attention = torch.full(
            [batch_size, seq_len, seq_len], False)  # (bs, SeqLen, SeqLen) filled with False
        decoder_padding_mask_cross_attention = torch.full(
            [batch_size, seq_len, seq_len], False)  # (bs, SeqLen, SeqLen) filled with False

        for idx in range(batch_size):
            # each sentence is tokenize character per character
            # Hence the number of token to build the mask is the length of the sentence
            eng_sentence_length, kn_sentence_length = len(eng_batch[idx]), len(kn_batch[idx])
            eng_chars_to_padding_mask = np.arange(eng_sentence_length + 1, seq_len)  # [56,57,....80]
            kn_chars_to_padding_mask = np.arange(kn_sentence_length + 1, seq_len)  # [21,22,23...80]
            # keep the top-left quadrant of the matrix. Assign the three other quadant to True
            encoder_padding_mask[idx, :, eng_chars_to_padding_mask] = True  # set the right columns to True
            encoder_padding_mask[idx, eng_chars_to_padding_mask, :] = True  # set the bottom rows to True
            # keep the top-left quadrant of the matrix. Assign the three other quadant to True
            # set the right columns to True
            decoder_padding_mask_self_attention[idx, :, kn_chars_to_padding_mask] = True
            decoder_padding_mask_self_attention[idx, kn_chars_to_padding_mask, :] = True  # set the bottom rows to True
            # keep the top-left quadrant of the matrix. Assign the three other quadant to True
            decoder_padding_mask_cross_attention[idx, :,
                                                 eng_chars_to_padding_mask] = True  # set the right columns to True
            decoder_padding_mask_cross_attention[idx, kn_chars_to_padding_mask, :] = True  # set the bottom rows to True

        # The sensor has the shape (bs, SeqLen, SeqLen)
        # In each element/matrix of the batch, (SeqLen, SeqLen matrix the top-left quadrant is set to 0 and the other 3 quadrant to -inf
        # In each element/matrix of the batch, the "size of the quadrant" in each matrix dependent on the length of the src sentence.
        encoder_self_attention_mask = torch.where(encoder_padding_mask, NEG_INFTY, 0)  # (bs, SeqLen, SeqLen)

        # The sensor has the shape (bs, SeqLen, SeqLen)
        # In each element/matrix of the batch, (SeqLen, SeqLen matrix the top-left quadrant is set to 0 and the other 3 quadrant to -inf
        # In each element/matrix of the batch, the "size of the quadrant" in each matrix dependent on the length of the tgt sentence.
        # Finally any cell above the diagonal will be set to -inf and the one bellow to 0
        # JEB: Understand the +, looks like on each cell bellow the diagonal mix between 0 (int) and bolean (False)
        decoder_self_attention_mask = torch.where(
            look_ahead_mask + decoder_padding_mask_self_attention, NEG_INFTY, 0)  # (bs, SeqLen, SeqLen)
        # JEB: Seems to be working
        # print(decoder_self_attention_mask[0])

        # The sensor has the shape (bs, SeqLen, SeqLen)
        # In each element/matrix of the batch, (SeqLen, SeqLen matrix the top-left quadrant is set to 0 and the other 3 quadrant to -inf
        # In each element/matrix of the batch, the "size of the quadrant" in each matrix dependent on the length of the tgt sentence.
        decoder_cross_attention_mask = torch.where(
            decoder_padding_mask_cross_attention, NEG_INFTY, 0)  # (bs, SeqLen, SeqLen)
        return encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask


def is_valid_tokens(sentence, vocab):
    for token in list(set(sentence)):
        if token not in vocab:
            print(bytes(token,"utf-8"))
            return False
    return True


def is_valid_length(sentence, max_sequence_length):
    return len(list(sentence)) < (max_sequence_length - 1)  # need to re-add the end token so leaving 1 space


def get_all_sentences6(ds, lang_idx):
    for item in ds:
        yield item[lang_idx]


def get_or_build_tokenizer6(config: dict, model_folder: str, ds, lang: str) -> Tokenizer:
    tokenizer_path = Path(model_folder + "/" + config['tokenizer_file'].format(lang) + ".json")
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(BPE(char_level=True, unk_token=UNK))
        # tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False,use_regex=False)
        # tokenizer.decoder = decoders.ByteLevel()
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(special_tokens=[UNK, PAD, SOS, EOS, ' ', '?', '!'], max_token_length=1, min_frequency=1)
        lang_idx = 0 if lang == "en" else 1
        tokenizer.train_from_iterator(get_all_sentences6(ds, lang_idx), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_tokenizer6(config: dict, model_folder: str, lang: str) -> Tokenizer:
    tokenizer_path = Path(model_folder + "/" + config['tokenizer_file'].format(lang) + ".json")
    if not Path.exists(tokenizer_path):
        print(f"Tokenizer does not exists {tokenizer_path}")
        raise ValueError(f"{tokenizer_path} Tokenizer does not exist")
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

# def get_ds6(config: dict, model_folder: str) -> Tuple[DataLoader, DataLoader, Tokenizer, Tokenizer]:


def get_ds6(config: dict, model_folder: str) -> Tuple[DataLoader, DataLoader, int, int, dict, dict, dict]:

    src_file = f"custom_datasets/{config['datasource']}_{config['lang_src']}_{config['lang_tgt']}/{config['lang_src']}.txt"
    tgt_file = f"custom_datasets/{config['datasource']}_{config['lang_src']}_{config['lang_tgt']}/{config['lang_tgt']}.txt"

    with open(src_file, 'r') as file:
        src_sentences = file.readlines()
    with open(tgt_file, 'r') as file:
        tgt_sentences = file.readlines()

    # Limit Number of sentences
    TOTAL_SENTENCES = 200000
    src_sentences = src_sentences[:TOTAL_SENTENCES]
    tgt_sentences = tgt_sentences[:TOTAL_SENTENCES]
    src_sentences = [sentence.rstrip('\n').lower() for sentence in src_sentences]
    tgt_sentences = [sentence.rstrip('\n') for sentence in tgt_sentences]

    full_ds = Dataset6(src_sentences, tgt_sentences)
    tokenizer_src = get_or_build_tokenizer6(config, model_folder, full_ds, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer6(config, model_folder, full_ds, config['lang_tgt'])

    valid_sentence_indicies = []
    for index in range(len(tgt_sentences)):
        tgt_sentence, src_sentence = tgt_sentences[index], src_sentences[index]
        if is_valid_length(tgt_sentence, config['seq_len']) \
                and is_valid_length(src_sentence, config['seq_len']):
            if is_valid_tokens(tgt_sentence, tokenizer_tgt.get_vocab()) \
                    and is_valid_tokens(src_sentence, tokenizer_src.get_vocab()):
                valid_sentence_indicies.append(index)
            else:
                # print("Unknwon token")
                pass
        else:
            # print("Invalid length")
            pass

    print(f"Number of sentences: {len(tgt_sentences)}")
    print(f"Number of valid sentences: {len(valid_sentence_indicies)}")

    tgt_sentences = [tgt_sentences[i] for i in valid_sentence_indicies]
    src_sentences = [src_sentences[i] for i in valid_sentence_indicies]

    # print(tgt_sentences[:3])

    ds_raw = Dataset6(src_sentences, tgt_sentences)

    index_to_tgt = {v: k for i, (k, v) in enumerate(tokenizer_tgt.get_vocab().items())}

    # keep 90% for training and 10% for validation
    # train_ds_size = int(0.9 * len(ds_raw))
    # val_ds_size = len(ds_raw) - train_ds_size
    # train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    # train_ds = Dataset6(train_ds_raw, tokenizer_src, tokenizer_tgt,
    #                     config['lang_src'], config['lang_tgt'], config['seq_len'])
    # val_ds = Dataset6(val_ds_raw, tokenizer_src, tokenizer_tgt,
    #                   config['lang_src'], config['lang_tgt'], config['seq_len'])

    train_dataloader = DataLoader(ds_raw, config['batch_size'])

    # return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt
    return train_dataloader, None, len(tokenizer_src.get_vocab()), len(tokenizer_tgt.get_vocab()), tokenizer_src.get_vocab(), tokenizer_tgt.get_vocab(), index_to_tgt


def get_testing_ds6(config: dict, model_folder: str, sentence: str) -> Tuple[str, str, Tokenizer, Tokenizer]:
    return sentence, "", None, None
