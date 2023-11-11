#!/usr/bin/env python3
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from tokenizers import Tokenizer
import torchmetrics
import torchmetrics.text

# from torchtext.data.utils import get_tokenizer

from pathlib import Path

from tqdm import tqdm
from dataset1 import get_ds1, casual_mask
from dataset2 import get_ds2
from dataset3 import get_ds3

from config import get_model_folder, get_weights_file_path, get_config, latest_weights_file_path
from config import get_console_width, get_device

from model1 import Transformer1, build_transformer1
from model2 import Transformer2, build_transformer2
from model3 import Transformer3, build_transformer3
from model4 import Transformer4, build_transformer4
from model5 import Transformer5, build_transformer5
from model6 import Transformer6, build_transformer6

from dataset1 import get_ds1
from dataset2 import get_ds2
from dataset3 import get_ds3
from dataset6 import get_ds6


def reload_model(config, model, optimizer, initial_epoch, global_step):
    preload = config['preload']
    model_filename = latest_weights_file_path(
        config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])  # JEB: This was not in the video
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')
    return model, initial_epoch, optimizer, global_step


def save_model(model, optimizer, epoch, global_step):
    # Save the model at the end of every epoch
    model_filename = get_weights_file_path(config, f'{epoch:02d}')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'global_step': global_step
    }, model_filename)


def build_model1(config: dict, vocab_src_len: int, vocab_tgt_len: int) -> Transformer1:
    model = build_transformer1(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'],
                               d_model=config['d_model'], N=config['N'], h=config['h'], dropout=config['dropout'], d_ff=config['d_ff'])
    return model


def build_model2(config: dict, vocab_src_len: int, vocab_tgt_len: int) -> Transformer2:
    model = build_transformer2(vocab_src_len, vocab_tgt_len, config['seq_len'],
                               d_model=config['d_model'], N=config['N'], h=config['h'], dropout=config['dropout'], d_ff=config['d_ff'])
    return model


def build_model3(config: dict, vocab_src_len: int, vocab_tgt_len: int) -> Transformer3:
    model = build_transformer3(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'],
                               d_model=config['d_model'], n_layers=config['N'], heads=config['h'], dropout=config['dropout'])
    return model


def build_model4(config: dict, vocab_src_len: int, vocab_tgt_len: int) -> Transformer4:
    model = build_transformer4(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'],
                               d_model=config['d_model'], N=config['N'], h=config['h'], dropout=config['dropout'], d_ff=config['d_ff'])
    return model


def build_model5(config: dict, vocab_src_len: int, vocab_tgt_len: int) -> Transformer5:
    model = build_transformer5(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'],
                               d_model=config['d_model'], N=config['N'], h=config['h'], dropout=config['dropout'], d_ff=config['d_ff'])
    return model


def build_model6(config: dict, vocab_src_len: int, vocab_tgt_len: int, src_to_index: dict, tgt_to_index: dict) -> Transformer6:
    model = build_transformer6(vocab_src_len, vocab_tgt_len, src_to_index, tgt_to_index, config['seq_len'], config['seq_len'],
                               d_model=config['d_model'], N=config['N'], h=config['h'], dropout=config['dropout'], d_ff=config['d_ff'])
    return model


def test_model1(config: dict, device):
    config['model'] = "model1"
    config['datasource'] = "opus_books"
    config['lang_src'] = "en"
    config['lang_tgt'] = "it"

    model_folder = get_model_folder(config)
    Path(model_folder).mkdir(parents=True, exist_ok=True)

    # train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds1(config, model_folder)
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds1(config, model_folder)
    model = build_model1(config, 500, 500).to(device)

    print(model)
    model.train()


def test_model2(config: dict, device):
    config['model'] = "model2"
    config['datasource'] = "translate"
    config['lang_src'] = "en"
    config['lang_tgt'] = "fr"

    model_folder = get_model_folder(config)
    Path(model_folder).mkdir(parents=True, exist_ok=True)

    # train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds1(config, model_folder)
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds2(config, model_folder)
    model = build_model2(config, 500, 500).to(device)

    print(model)
    model.train()


def test_model3(config: dict, device):
    config['model'] = "model3"
    config['datasource'] = "translate"
    config['lang_src'] = "en"
    config['lang_tgt'] = "fr"

    model_folder = get_model_folder(config)
    Path(model_folder).mkdir(parents=True, exist_ok=True)

    # train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds1(config, model_folder)
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds3(config, model_folder)
    model = build_model3(config, 500, 500).to(device)

    print(model)
    model.train()


def test_model4(config: dict, device):
    config['model'] = "model4"
    config['datasource'] = "translate"
    config['lang_src'] = "en"
    config['lang_tgt'] = "fr"

    model_folder = get_model_folder(config)
    Path(model_folder).mkdir(parents=True, exist_ok=True)

    # train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds2(config, model_folder)
    model = build_model4(config, 500, 500).to(device)

    print(model)
    model.train()


def test_model5(config: dict, device):
    config['model'] = "model5"
    config['datasource'] = "translate"
    config['lang_src'] = "en"
    config['lang_tgt'] = "fr"

    model_folder = get_model_folder(config)
    Path(model_folder).mkdir(parents=True, exist_ok=True)

    # train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds2(config, model_folder)
    model = build_model5(config, 500, 500).to(device)

    print(model)
    model.train()


def test_model6(config: dict, device):
    config['model'] = "model6"
    config['datasource'] = "translate"
    config['lang_src'] = "en"
    config['lang_tgt'] = "kn"

    model_folder = get_model_folder(config)
    Path(model_folder).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, src_vocab_size, tgt_vocab_size, src_to_index, tgt_to_index, index_to_tgt = get_ds6(
        config, model_folder)
    model = build_model6(config, src_vocab_size, tgt_vocab_size, src_to_index, tgt_to_index).to(device)

    print(model)
    model.train()


if __name__ == '__main__':
    # warnings.filterwarnings('ignore')
    config = get_config()
    device = get_device()

    test_model1(config, device)
    test_model2(config, device)
    test_model3(config, device)
    test_model4(config, device)
    test_model5(config, device)
    test_model6(config, device)
