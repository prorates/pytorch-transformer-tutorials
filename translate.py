#!/usr/bin/env python3
from pathlib import Path
from config import get_model_folder, get_weights_file_path, get_config, latest_weights_file_path, get_device
from model1 import build_transformer1
from model1 import Transformer1
from tokenizers import Tokenizer
from datasets import load_dataset
from dataset1 import Dataset1, translation_mask
import torch
import sys


def greedy_decode(model: Transformer1, source, source_mask, tokenizer_src: Tokenizer,
                  tokenizer_tgt: Tokenizer, max_len: int, device):
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')
    sos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    # JEB: source at point is not a batch hence the unsqueeze(0).
    # JEB: not sure the is the only place it is needeed
    encoder_output = model.encode(source.unsqueeze(0), source_mask)

    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)

    # Generate the translation word by word
    while decoder_input.size(1) < max_len:
        # build mask for target and calculate output
        decoder_mask = translation_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculate the output of the decoder
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # project next token
        prob = model.project(out[:, -1])

        # Select the token with the max probability (because it is a greedy search)
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        # print the translated word
        print(f"{tokenizer_tgt.decode([next_word.item()])}", end=' ')

        # break if we predict the end of sentence token
        if next_word == eos_idx:
            break

    return decoder_input[0].tolist()


def run_translation(label: str, sentence: str, model: Transformer1, tokenizer_src: Tokenizer,
                    tokenizer_tgt: Tokenizer, max_len: int, device):
    sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
    eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
    pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    model.eval()
    with torch.no_grad():
        # Precompute the encoder output and reuse it for every generation step
        # source = tokenizer_src.encode(sentence)
        # source = torch.cat([
        #     torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64),
        #     torch.tensor(source.ids, dtype=torch.int64),
        #     torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64),
        #     torch.tensor([tokenizer_src.token_to_id('[PAD]')] * (max_len - len(source.ids) - 2), dtype=torch.int64)
        # ], dim=0).to(device)
        enc_input_tokens = tokenizer_src.encode(sentence).ids
        enc_num_padding_tokens = max_len - len(enc_input_tokens) - 2
        source = torch.cat(
            [
                sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                eos_token,
                torch.tensor([pad_token] * enc_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        ).to(device)
        source_mask = (source != pad_token.to(device)).unsqueeze(0).unsqueeze(0).int().to(device)
        # assert source.size(0) == 1, "Batch size must be 1 for validation"

        # Print the source sentence and target start prompt
        print(f"{f'SOURCE: ':>12}{sentence}")
        if label != "":
            print(f"{f'TARGET: ':>12}{label}")
        print(f"{f'PREDICTED: ':>12}", end='')

        model_out = greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device)

    # convert ids to tokens
    return tokenizer_tgt.decode(model_out)


def get_tokenizer(config: dict, model_folder: str, lang: str) -> Tokenizer:
    tokenizer_path = Path(model_folder + "/" + config['tokenizer_file'].format(lang) + ".json")
    if not Path.exists(tokenizer_path):
        print(f"Tokenizer does not exists {tokenizer_path}")
        raise ValueError(f"{tokenizer_path} Tokenizer does not exist")
    else:
        return Tokenizer.from_file(str(tokenizer_path))


def get_model(config: dict, vocab_src_len: int, vocab_tgt_len: int) -> Transformer1:
    model = build_transformer1(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'],
                               d_model=config['d_model'], N=config['N'], h=config['h'], dropout=config['dropout'], d_ff=config['d_ff'])
    return model


def translate(config: dict, sentence: str):
    device = get_device()

    model_folder = get_model_folder(config)
    if not Path.exists(Path(model_folder)):
        raise ValueError(f"{model_folder} model_folder does not exist")

    tokenizer_src = get_tokenizer(config, model_folder, config['lang_src'])
    tokenizer_tgt = get_tokenizer(config, model_folder, config['lang_tgt'])
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    # Load the pretrained weights
    preload = config['preload']
    model_filename = latest_weights_file_path(
        config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])  # JEB: This was not in the video
    else:
        raise ValueError(f"No pretrained model to load")

    # if the sentence is a number use it as an index to the test set
    label = ""
    if type(sentence) == int or sentence.isdigit():
        id = int(sentence)
        ds = load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split='all')
        ds = Dataset1(ds, tokenizer_src, tokenizer_tgt,
                      config['lang_src'], config['lang_tgt'], config['seq_len'])
        sentence = ds[id]['src_text']
        label = ds[id]["tgt_text"]

    run_translation(label, sentence, model, tokenizer_src, tokenizer_tgt, config['seq_len'], device)


if __name__ == '__main__':
    # warnings.filterwarnings('ignore')
    config = get_config()
    # read sentence from argument
    translate(config, sys.argv[1] if len(sys.argv) > 1 else "I am not a very good a student.")
