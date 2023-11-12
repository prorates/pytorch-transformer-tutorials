import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import (get_console_width, get_device, get_model_folder, get_config)
from dataset7 import get_ds7
from model7 import Transformer7, build_transformer7
from utils import reload_model, save_model


def build_model7(config: dict, vocab_tgt_len: int) -> Transformer7:
    model = build_transformer7(vocab_tgt_len,
                               d_model=config['d_model'], N=config['N'], h=config['h'], dropout=config['dropout'], d_ff=config['d_ff'])
    return model


def train_model7(config: dict):
    device = get_device()

    model_folder = get_model_folder(config)
    Path(model_folder).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, test_dataloader, tokenizer_tgt = get_ds7(config, model_folder)
    transformer = build_model7(config, tokenizer_tgt.get_vocab_size()).to(device)

    # Tensorboard
    writer = SummaryWriter(get_model_folder(config) + "/" + config['experiment_name'])

    lr = 5.0  # learning rate
    optimizer = torch.optim.SGD(transformer.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    best_val_loss = float('inf')
    total_loss = 0
    initial_epoch = 0
    global_step = 0
    log_interval = 200

    transformer, initial_epoch, optimizer, global_step = reload_model(
        config, transformer, optimizer, initial_epoch, global_step)
    loss_fn = nn.CrossEntropyLoss()

    console_width = get_console_width()

    start_time = time.time()
    num_batches = len(train_dataloader)

    for epoch in range(initial_epoch, config['num_epochs']):
        epoch_start_time = time.time()
        if (device == 'cuda'):
            torch.cuda.empty_cache()

        transformer.train()  # moved inside for run_validation at each step
        batch_iterator = tqdm(train_dataloader, desc=f'Processing epoch {epoch:02d}')
        for batch_num, batch in enumerate(batch_iterator):
            data, targets = batch
            data = data.squeeze(0).to(device)
            targets = targets.squeeze(0).to(device)
            # data: Tensor, shape ``[seq_len, batch_size]``
            # src_mask: Tensor, shape ``[seq_len, seq_len]``
            # output Tensor of shape ``[seq_len, batch_size, ntoken]``
            output = transformer(data)
            output_flat = output.view(-1, tokenizer_tgt.get_vocab_size())
            loss = loss_fn(output_flat, targets)
            batch_iterator.set_postfix({"Loss": f"{loss.item():6.3f}"})

            # Log of loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(transformer.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()
            if batch_num % log_interval == 0 and batch_num > 0:
                lr = scheduler.get_last_lr()[0]
                ms_per_batch = (time.time() - start_time) * 1000 / log_interval
                cur_loss = total_loss / log_interval
                ppl = math.exp(cur_loss)
                batch_iterator.write(
                    f'| epoch {epoch:3d} | {batch_num:5d}/{num_batches:5d} batches | ' f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | ' f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
                total_loss = 0
                start_time = time.time()

            global_step += 1

        # Run validation at the end of each epoch
        val_loss = float(0)
        val_loss = evaluate_model7(transformer, val_dataloader, tokenizer_tgt.get_vocab_size(), device)

        val_ppl = math.exp(val_loss)
        elapsed = time.time() - epoch_start_time
        batch_iterator.write('-' * console_width)
        batch_iterator.write(
            f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | ' f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
        batch_iterator.write('-' * console_width)

        best_model_yet = False
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_yet = True

        # Save the model at the end of every epoch
        save_model(config, transformer, optimizer, epoch, global_step, best_model_yet)

        #
        scheduler.step()

    # test_loss = evaluate(model, test_data)
    # test_ppl = math.exp(test_loss)
    # print(f'| End of training | test loss {test_loss:5.2f} | ' f'test ppl {test_ppl:8.2f}')


def evaluate_model7(transformer: Transformer7, validation_ds: DataLoader, ntokens: int, device):

    transformer.eval()  # turn on evaluation mode
    total_loss = 0.

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_num, batch in enumerate(validation_ds):
            # count += 1
            data, targets = batch
            data = data.squeeze(0).to(device)
            targets = targets.squeeze(0).to(device)

            seq_len = data.size(0)
            output = transformer(data)
            output_flat = output.view(-1, ntokens)
            total_loss += seq_len * criterion(output_flat, targets).item()

            # Print the message to the console without interfering with the progress bar
            # print_msg('-' * console_width)

            # if count == num_examples:
            #    print_msg('-' * console_width)
            #    break

    return total_loss / (len(validation_ds) - 1)

def translate7(config: dict, sentence: str):
    device = get_device()

    model_folder = get_model_folder(config)
    if not Path.exists(Path(model_folder)):
        raise ValueError(f"{model_folder} model_folder does not exist")

    raise RuntimeError("Not implemented yet")

def test_model7(config: dict, device):
    config['model'] = "model7"
    config['datasource'] = "translate"
    config['lang_src'] = "en"
    config['lang_tgt'] = "fr"

    model_folder = get_model_folder(config)
    Path(model_folder).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, test_dataloader, tokenizer_tgt = get_ds7(config, model_folder)
    model = build_model7(config, tokenizer_tgt.get_vocab_size()).to(device)

    print(model)
    model.train()

if __name__ == '__main__':
    # warnings.filterwarnings('ignore')
    config = get_config()
    device = get_device()
    test_model7(config, device)
