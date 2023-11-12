#!/usr/bin/env python3
import sys
import getopt
import time
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from tokenizers import Tokenizer
import torchmetrics
import torchmetrics.text

from tqdm import tqdm
from dataset1 import get_ds1, casual_mask
from dataset2 import get_ds2
from dataset3 import get_ds3
from dataset6 import get_ds6, Dataset6
from dataset7 import get_ds7

from config import get_model_folder, get_weights_file_path, get_config, latest_weights_file_path, get_best_model_params_path
from config import get_console_width, get_device
from config import EOS, SOS, PAD, UNK

from model1 import Transformer1, build_transformer1
from model2 import Transformer2, build_transformer2
from model3 import Transformer3, build_transformer3
from model4 import Transformer4, build_transformer4
from model5 import Transformer5, build_transformer5
from model6 import Transformer6, build_transformer6
from model7 import Transformer7, build_transformer7


def collect_training_metrics(writer, predicted, expected, global_step):
    if writer:
        # Evaluate the character error rate
        # Compute the char error rate
        metric = torchmetrics.text.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar('validation cer', cer, global_step)
        writer.flush()

        # Compute the word error rate
        metric = torchmetrics.text.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar('validation wer', wer, global_step)
        writer.flush()

        # Compute the BLEU metric
        metric = torchmetrics.text.BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar('validation BLEU', bleu, global_step)
        writer.flush()


def reload_model(config, model, optimizer, initial_epoch, global_step):
    preload = config['preload']
    model_filename = latest_weights_file_path(
        config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])  # JEB: This was not in the vide
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')
    return model, initial_epoch, optimizer, global_step


def save_model(config, model, optimizer, epoch: int, global_step: int, best_model_yet: bool = False):
    # Save the model at the end of every epoch
    model_filename = get_weights_file_path(config, f'{epoch:02d}')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'global_step': global_step
    }, model_filename)

    if (best_model_yet):
        best_model_filename = get_best_model_params_path(config, f'{epoch:02d}')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, best_model_filename)


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


def build_model7(config: dict, vocab_tgt_len: int) -> Transformer7:
    model = build_transformer7(vocab_tgt_len,
                               d_model=config['d_model'], N=config['N'], h=config['h'], dropout=config['dropout'], d_ff=config['d_ff'])
    return model


def validate_model1(model: Transformer1, validation_ds: DataLoader, tokenizer_src: Tokenizer, tokenizer_tgt: Tokenizer,
                    max_len: int, device, print_msg, global_step: int, writer, num_examples: int = 2):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    console_width = get_console_width()

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)

            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            eos_idx = tokenizer_tgt.token_to_id(EOS)
            sos_idx = tokenizer_tgt.token_to_id(SOS)
            # JEB: Not sure this is really consistent
            # encoder_input has shape (bs=1, SeqLen)
            # encoder_mask has shape (bs=1, 1, 1, SeqLen)
            # model_out has shape (SeqLen)
            model_out = model.greedy_decode(encoder_input, encoder_mask, sos_idx, eos_idx, max_len, device)

            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            # Print the message to the console without interfering with the progress bar
            print_msg('-' * console_width)
            print_msg(f"{f'Source: ':>15}{source_text}")
            print_msg(f"{f'Target: ':>15}{target_text}")
            print_msg(f"{f'Prediction: ':>15}{model_out_text}")

            if count == num_examples:
                print_msg('-' * console_width)
                break
    if writer:
        collect_training_metrics(writer, predicted, expected, global_step)


def train_model1(config: dict):
    device = get_device()

    model_folder = get_model_folder(config)
    Path(model_folder).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds1(config, model_folder)
    model = build_model1(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    # Tensorboard
    writer = SummaryWriter(get_model_folder(config) + "/" + config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    initial_epoch = 0
    global_step = 0
    model, initial_epoch, optimizer, global_step = reload_model(config, model, optimizer, initial_epoch, global_step)

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id(PAD), label_smoothing=0.1).to(device)

    console_width = get_console_width()

    for epoch in range(initial_epoch, config['num_epochs']):
        if (device == 'cuda'):
            torch.cuda.empty_cache()

        model.train()  # moved inside for run_validation at each step
        batch_iterator = tqdm(train_dataloader, desc=f'Processing epoch {epoch:02d}')
        for batch_num, batch in enumerate(batch_iterator):

            encoder_input = batch['encoder_input'].to(device)  # (B, SeqLen)
            decoder_input = batch['decoder_input'].to(device)  # (B, SeqLen)
            encoder_mask = batch['encoder_mask'].to(device)  # (B, 1, 1, SeqLen)
            decoder_mask = batch['decoder_mask'].to(device)  # (B, 1, SeqLen, SeqLen)

            # Run the tensors through the transformer
            encoder_output = model.encode(encoder_input, encoder_mask)  # (B, SeqLen, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input,
                                          decoder_mask)  # (B, SeqLen, d_model)
            proj_output = model.project(decoder_output)  # (B, SeqLen, tgt_vocab_size)

            # Compare the output with the label
            label = batch['label'].to(device)  # (B, SeqLen)

            # (B, SeqLen, tgt_vocab_size) --> (B * SeqLen, tgt_vocab_size)
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"Loss": f"{loss.item():6.3f}"})

            # Log of loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # backpropagate the loss
            loss.backward()

            # update the weights
            optimizer.step()

            if (batch_num > 0) and (batch_num % 100 == 0):
                batch_iterator.write('-' * console_width)
                batch_iterator.write(f"{f'Source: ':>15}{batch['src_text'][0]}")
                batch_iterator.write(f"{f'Target: ':>15}{batch['tgt_text'][0]}")
                kn_sentence_predicted = torch.argmax(proj_output[0], axis=1)
                # JEB: Figure out how to get decode to stop at eos
                # predicted_sentence = tokenizer_tgt.decode(kn_sentence_predicted.detach().cpu().numpy(), skip_special_tokens=True)
                predicted_words = []
                for idx in kn_sentence_predicted:
                    if idx == tokenizer_tgt.token_to_id(EOS):
                        break
                    predicted_words.append(tokenizer_tgt.id_to_token(idx.item()))
                predicted_sentence = ' '.join(predicted_words)
                batch_iterator.write(f"{f'Prediction: ':>15}{predicted_sentence}")
                batch_iterator.write('-' * console_width)

            # Initialize to None instead of 0. Supposed to provide better performance.
            # optimizer.zero_grad()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        # Run validation at the end of each epoch
        validate_model1(model, val_dataloader, tokenizer_src, tokenizer_tgt,
                        config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)

        # Save the model at the end of every epoch
        save_model(config, model, optimizer, epoch, global_step)


def train_model2(config: dict):
    device = get_device()

    model_folder = get_model_folder(config)
    Path(model_folder).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds2(config, model_folder)
    model = build_model2(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    # Tensorboard
    writer = SummaryWriter(get_model_folder(config) + "/" + config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], betas=(0.9, 0.98), eps=1e-9)

    initial_epoch = 0
    global_step = 0
    model, initial_epoch, optimizer, global_step = reload_model(config, model, optimizer, initial_epoch, global_step)

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id(PAD)).to(device)

    console_width = get_console_width()

    for epoch in range(initial_epoch, config['num_epochs']):
        if (device == 'cuda'):
            torch.cuda.empty_cache()

        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f'Processing epoch {epoch:02d}')
        for batch_num, batch in enumerate(batch_iterator):
            optimizer.zero_grad()
            src_data = batch['src'].to(device)  # (B, SeqLen)
            tgt_data = batch['tgt'].to(device)  # (B, SeqLen)
            src_mask = batch['src_mask'].to(device)  # (B, 1, 1, SeqLen)
            tgt_mask = batch['tgt_mask'].to(device)  # (B, 1, 1, SeqLen)

            # JEB: Like for Model3. Need to have the full length
            # output = model(src_data, tgt_data[:, :-1].to(device), src_mask, tgt_mask)
            output = model(src_data, tgt_data, src_mask, tgt_mask)

            # JEB: Same issue as for Model3
            # loss = loss_fn(output.contiguous().view(-1, tokenizer_tgt.get_vocab_size()),
            #                 tgt_data[:, 1:].contiguous().view(-1))
            loss = loss_fn(output.contiguous().view(-1, tokenizer_tgt.get_vocab_size()),
                           tgt_data.contiguous().view(-1))
            batch_iterator.set_postfix({"Loss": f"{loss.item():6.3f}"})

            # Log of loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # backpropagate the loss
            loss.backward()

            # update the weights
            optimizer.step()

            if (batch_num > 0) and (batch_num % 100 == 0):
                batch_iterator.write('-' * console_width)
                batch_iterator.write(f"{f'Source: ':>15}{batch['src_text'][0]}")
                batch_iterator.write(f"{f'Target: ':>15}{batch['tgt_text'][0]}")
                kn_sentence_predicted = torch.argmax(output[0], axis=1)
                # JEB: Figure out how to get decode to stop at eos
                # predicted_sentence = tokenizer_tgt.decode(kn_sentence_predicted.detach().cpu().numpy(), skip_special_tokens=True)
                predicted_words = []
                for idx in kn_sentence_predicted:
                    if idx == tokenizer_tgt.token_to_id(EOS):
                        break
                    predicted_words.append(tokenizer_tgt.id_to_token(idx.item()))
                predicted_sentence = ' '.join(predicted_words)
                batch_iterator.write(f"{f'Prediction: ':>15}{predicted_sentence}")
                batch_iterator.write('-' * console_width)

            global_step += 1
            # print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

        # Save the model at the end of every epoch
        save_model(config, model, optimizer, epoch, global_step)


def validate_model3(model: Transformer3, validation_ds: DataLoader, tokenizer_src: Tokenizer, tokenizer_tgt: Tokenizer,
                    max_len: int, device, print_msg, global_step: int, writer, num_examples: int = 2):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    console_width = get_console_width()

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch['src'].to(device)
            encoder_mask = batch['src_mask'].to(device)

            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            # model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            # Print the message to the console without interfering with the progress bar
            print_msg('-' * console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-' * console_width)
                break
    if writer:
        collect_training_metrics(writer, predicted, expected, global_step)


def train_model3(config: dict):
    device = get_device()

    model_folder = get_model_folder(config)
    Path(model_folder).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds3(config, model_folder)
    model = build_model3(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], betas=(0.9, 0.98), eps=1e-9)
    # if opt.SGDR == True:
    #     opt.sched = CosineWithRestarts(opt.optimizer, T_max=opt.train_len)

    # Tensorboard
    writer = SummaryWriter(get_model_folder(config) + "/" + config['experiment_name'])

    initial_epoch = 0
    global_step = 0
    model, initial_epoch, optimizer, global_step = reload_model(config, model, optimizer, initial_epoch, global_step)

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id(PAD), label_smoothing=0.1).to(device)

    console_width = get_console_width()

    for epoch in range(initial_epoch, config['num_epochs']):
        if (device == 'cuda'):
            torch.cuda.empty_cache()

        model.train()  # moved inside for run_validation at each step

        total_loss = 0

        batch_iterator = tqdm(train_dataloader, desc=f'Processing epoch {epoch:02d}')
        for batch_num, batch in enumerate(batch_iterator):

            src = batch['src'].to(device)  # (B, SeqLen)
            trg = batch['trg'].to(device)  # (B, SeqLen)
            src_mask = batch['src_mask'].to(device)  # (B, 1, 1, SeqLen)
            trg_mask = batch['trg_mask'].to(device)  # (B, 1, SeqLen, SeqLen)
            label = batch['label'].to(device)  # (B, SeqLen)

            # src = batch.src.transpose(0, 1).to(device)
            # trg = batch.trg.transpose(0, 1).to(device)
            # trg_input = trg[:, :-1]
            # src_mask, trg_mask = create_masks(src, trg_input, opt)
            # src_mask.to(device)
            # trg_mask.to(device)

            # JEB: Mask computation is different. No need to remove last one
            # trg_input = trg[:, :-1]
            # preds = model(src, trg_input, src_mask, trg_mask)
            preds = model(src, trg, src_mask, trg_mask)

            # JEB: Mask computation is different. No need to remove last one
            # ys = trg[:, 1:].contiguous().view(-1)
            ys = trg.contiguous().view(-1)

            optimizer.zero_grad()
            # JEB: Use the torch method instead
            # loss = F.cross_entropy(preds.view(-1, preds.size(-1)), ys, ignore_index=opt.trg_pad)
            loss = loss_fn(preds.view(-1, preds.size(-1)), ys)
            batch_iterator.set_postfix({"Loss": f"{loss.item():6.3f}"})

            loss.backward()
            optimizer.step()
            # if opt.SGDR == True:
            #    opt.sched.step()

            if (batch_num > 0) and (batch_num % 100 == 0):
                batch_iterator.write('-' * console_width)
                batch_iterator.write(f"{f'Source: ':>15}{batch['src_text'][0]}")
                batch_iterator.write(f"{f'Target: ':>15}{batch['tgt_text'][0]}")
                kn_sentence_predicted = torch.argmax(preds[0], axis=1)
                # JEB: Figure out how to get decode to stop at eos
                # predicted_sentence = tokenizer_tgt.decode(kn_sentence_predicted.detach().cpu().numpy(), skip_special_tokens=True)
                predicted_words = []
                for idx in kn_sentence_predicted:
                    if idx == tokenizer_tgt.token_to_id(EOS):
                        break
                    predicted_words.append(tokenizer_tgt.id_to_token(idx.item()))
                predicted_sentence = ' '.join(predicted_words)
                batch_iterator.write(f"{f'Prediction: ':>15}{predicted_sentence}")
                batch_iterator.write('-' * console_width)

            total_loss += loss.item()

        # Save the model at the end of every epoch
        save_model(config, model, optimizer, epoch, global_step)


def train_model4(config: dict):
    device = get_device()

    model_folder = get_model_folder(config)
    Path(model_folder).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds3(config, model_folder)
    model = build_model4(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    raise RuntimeError("Training for model4 not implemented")


def train_model5(config: dict):
    device = get_device()

    model_folder = get_model_folder(config)
    Path(model_folder).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds3(config, model_folder)
    model = build_model5(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    raise RuntimeError("Training for model5 not implemented")


def train_model6(config: dict):
    device = get_device()

    model_folder = get_model_folder(config)
    Path(model_folder).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, src_vocab_size, tgt_vocab_size, src_to_index, tgt_to_index, index_to_tgt = get_ds6(
        config, model_folder)
    transformer = build_model6(config, src_vocab_size, tgt_vocab_size, src_to_index, tgt_to_index).to(device)

    # Tensorboard
    writer = SummaryWriter(get_model_folder(config) + "/" + config['experiment_name'])

    optimizer = torch.optim.Adam(transformer.parameters(), lr=config['lr'])

    total_loss = 0
    initial_epoch = 0
    global_step = 0

    transformer, initial_epoch, optimizer, global_step = reload_model(
        config, transformer, optimizer, initial_epoch, global_step)
    # loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id(PAD), reduction='none')
    loss_fn = nn.CrossEntropyLoss(ignore_index=tgt_to_index[PAD], reduction='none')

    console_width = get_console_width()

    for epoch in range(initial_epoch, config['num_epochs']):
        if (device == 'cuda'):
            torch.cuda.empty_cache()

        transformer.train()  # moved inside for run_validation at each step
        batch_iterator = tqdm(train_dataloader, desc=f'Processing epoch {epoch:02d}')
        for batch_num, batch in enumerate(batch_iterator):

            # src_batched_sentences: tuple[str], tgt_batched_sentences: tuple[str]
            src_batched_sentences, tgt_batched_sentences = batch
            encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask = Dataset6.create_masks(
                src_batched_sentences, tgt_batched_sentences, config['seq_len'])
            optimizer.zero_grad()
            predicted_tokens = transformer(src_batched_sentences,
                                           tgt_batched_sentences,
                                           encoder_self_attention_mask.to(device),
                                           decoder_self_attention_mask.to(device),
                                           decoder_cross_attention_mask.to(device),
                                           enc_start_token=False,  # During training, model6 does not add sos to encoder input
                                           enc_end_token=False,  # During training, model6 does not add sos to encoder input
                                           dec_start_token=True,  # During training, model6 DOES add sos to decoder input
                                           dec_end_token=True)  # During training, model6 DOES add eos to decoder input
            expected_tokens = transformer.decoder.sentence_embedding.batch_tokenize(
                tgt_batched_sentences, start_token=False, end_token=True)
            loss = loss_fn(predicted_tokens.view(-1, tgt_vocab_size).to(device),
                           expected_tokens.view(-1).to(device)).to(device)

            valid_indicies = torch.where(expected_tokens.view(-1) == tgt_to_index[PAD], False, True)
            loss = loss.sum() / valid_indicies.sum()
            batch_iterator.set_postfix({"Loss": f"{loss.item():6.3f}"})

            # Log of loss
            writer.add_scalar('train loss', loss, global_step)
            writer.flush()

            loss.backward()
            optimizer.step()

            # train_losses.append(loss.item())
            if (batch_num > 0) and (batch_num % 100 == 0):
                batch_iterator.write('-' * console_width)
                batch_iterator.write(f"{f'Source: ':>15}{src_batched_sentences[0]}")
                batch_iterator.write(f"{f'Target: ':>15}{tgt_batched_sentences[0]}")
                kn_sentence_predicted = torch.argmax(predicted_tokens[0], axis=1)
                predicted_sentence = ""
                for idx in kn_sentence_predicted:
                    if idx == tgt_to_index[EOS]:
                        break
                    predicted_sentence += index_to_tgt[idx.item()]
                batch_iterator.write(f"{f'Prediction: ':>15}{predicted_sentence}")
                batch_iterator.write('-' * console_width)

            # if batch_num % 20 == 0:
            #     validate_model6(transformer, val_dataloader, index_to_tgt,
            #                     config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)

        # Run validation at the end of each epoch
        validate_model6(transformer, val_dataloader, index_to_tgt,
                        config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)

        # Save the model at the end of every epoch
        save_model(config, transformer, optimizer, epoch, global_step)


def validate_model6(transformer: Transformer6, validation_ds: DataLoader, index_to_tgt: dict,
                    max_len: int, device, print_msg, global_step: int, writer, num_examples: int = 2):

    transformer.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    console_width = get_console_width()

    with torch.no_grad():
        for batch in validation_ds:
            count += 1

            # src_batched_sentences: tuple[str], tgt_batched_sentences: tuple[str]
            src_batched_sentences, expected_batched_sentences = batch
            assert len(src_batched_sentences) == 1, "Batch size must be 1 for validation"

            predicated_batched_sentences = transformer.greedy_decode(
                src_batched_sentences, max_len, index_to_tgt, device)

            source_texts.append(src_batched_sentences[0])
            expected.append(expected_batched_sentences[0])
            predicted.append(predicated_batched_sentences[0])

            # Print the message to the console without interfering with the progress bar
            print_msg('-' * console_width)
            print_msg(f"{f'Source: ':>15}{src_batched_sentences[0]}")
            print_msg(f"{f'Target: ':>15}{expected_batched_sentences[0]}")
            print_msg(f"{f'Prediction: ':>15}{predicated_batched_sentences[0]}")

            if count == num_examples:
                print_msg('-' * console_width)
                break
    if writer:
        collect_training_metrics(writer, predicted, expected, global_step)


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
        val_loss = validate_model7(transformer, val_dataloader, tokenizer_tgt.get_vocab_size(), device)

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


def validate_model7(transformer: Transformer7, validation_ds: DataLoader, ntokens: int, device):

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


def main(argv):
    config_filename = None
    model_folder = None
    try:
        opts, args = getopt.getopt(argv, "hc:m:", ["config=", "modelfolder="])
    except getopt.GetoptError:
        print('train.py -c <config_file> -m <model_folder>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('train.py -c <config_file> -m <model_folder>')
            sys.exit()
        elif opt in ("-c", "--config"):
            config_filename = arg
        elif opt in ("-m", "--modelfolder"):
            model_folder = arg

    # warnings.filterwarnings('ignore')
    config = get_config(config_filename, model_folder)

    match config['alt_model']:
        case "model1":
            train_model1(config)
        case "model2":
            train_model2(config)
        case "model3":
            train_model3(config)
        case "model4":
            train_model4(config)
        case "model5":
            train_model5(config)
        case "model6":
            train_model6(config)
        case "model7":
            train_model7(config)
        case _:
            train_model1(config)


if __name__ == "__main__":
    main(sys.argv[1:])
