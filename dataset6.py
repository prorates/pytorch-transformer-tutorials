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
from config import EOS, SOS, PAD, UNK
import numpy as np

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


class Dataset6(Dataset):

    def __init__(self, english_sentences, kannada_sentences):
        self.english_sentences = english_sentences
        self.kannada_sentences = kannada_sentences

    def __len__(self):
        return len(self.english_sentences)

    def __getitem__(self, idx):
        return self.english_sentences[idx], self.kannada_sentences[idx]


def is_valid_tokens(sentence, vocab):
    for token in list(set(sentence)):
        if token not in vocab:
            return False
    return True


def is_valid_length(sentence, max_sequence_length):
    return len(list(sentence)) < (max_sequence_length - 1)  # need to re-add the end token so leaving 1 space


def get_ds6(config: dict, model_folder: str) -> Tuple[DataLoader, DataLoader, int, int, dict, dict]:

    index_to_kannada = {k: v for k, v in enumerate(kannada_vocabulary)}
    kannada_to_index = {v: k for k, v in enumerate(kannada_vocabulary)}
    index_to_english = {k: v for k, v in enumerate(english_vocabulary)}
    english_to_index = {v: k for k, v in enumerate(english_vocabulary)}

    english_file = 'custom_datasets/translate_en_kn/english.txt'  # replace this path with appropriate one
    kannada_file = 'custom_datasets/translate_en_kn/kannada.txt'  # replace this path with appropriate one

    with open(english_file, 'r') as file:
        english_sentences = file.readlines()
    with open(kannada_file, 'r') as file:
        kannada_sentences = file.readlines()

    # Limit Number of sentences
    TOTAL_SENTENCES = 200000
    english_sentences = english_sentences[:TOTAL_SENTENCES]
    kannada_sentences = kannada_sentences[:TOTAL_SENTENCES]
    english_sentences = [sentence.rstrip('\n').lower() for sentence in english_sentences]
    kannada_sentences = [sentence.rstrip('\n') for sentence in kannada_sentences]

    print(english_sentences[:10])
    print(kannada_sentences[:10])

    PERCENTILE = 97
    print(f"{PERCENTILE}th percentile length Kannada: {np.percentile([len(x) for x in kannada_sentences], PERCENTILE)}")
    print(f"{PERCENTILE}th percentile length English: {np.percentile([len(x) for x in english_sentences], PERCENTILE)}")

    valid_sentence_indicies = []
    for index in range(len(kannada_sentences)):
        kannada_sentence, english_sentence = kannada_sentences[index], english_sentences[index]
        if is_valid_length(kannada_sentence, config['seq_len']) \
                and is_valid_length(english_sentence, config['seq_len']) \
                and is_valid_tokens(kannada_sentence, kannada_vocabulary):
            valid_sentence_indicies.append(index)

    print(f"Number of sentences: {len(kannada_sentences)}")
    print(f"Number of valid sentences: {len(valid_sentence_indicies)}")

    kannada_sentences = [kannada_sentences[i] for i in valid_sentence_indicies]
    english_sentences = [english_sentences[i] for i in valid_sentence_indicies]

    print(kannada_sentences[:3])

    dataset = Dataset6(english_sentences, kannada_sentences)
    len(dataset)
    print(dataset[1])

    # keep 90% for training and 10% for validation
    # train_ds_size = int(0.9 * len(ds_raw))
    # val_ds_size = len(ds_raw) - train_ds_size
    # train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    # train_ds = Dataset6(train_ds_raw, tokenizer_src, tokenizer_tgt,
    #                     config['lang_src'], config['lang_tgt'], config['seq_len'])
    # val_ds = Dataset6(val_ds_raw, tokenizer_src, tokenizer_tgt,
    #                   config['lang_src'], config['lang_tgt'], config['seq_len'])

    train_dataloader = DataLoader(dataset, config['batch_size'])

    # return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt
    return train_dataloader, None, len(english_vocabulary), len(kannada_vocabulary), english_to_index, kannada_to_index


def get_testing_ds6(config: dict, model_folder: str, sentence: str) -> Tuple[str, str, Tokenizer, Tokenizer]:
    return sentence, "", None, None
