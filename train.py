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
from dataset6 import get_ds6, Dataset6

from config import get_model_folder, get_weights_file_path, get_config, latest_weights_file_path
from config import get_console_width, get_device
from config import EOS, SOS, PAD, UNK

from model1 import Transformer1, build_transformer1
from model2 import Transformer2, build_transformer2
from model3 import Transformer3, build_transformer3
from model4 import Transformer4, build_transformer4
from model5 import Transformer5, build_transformer5
from model6 import Transformer6, build_transformer6


def greedy_decode(model: Transformer1, source, source_mask, tokenizer_src: Tokenizer,
                  tokenizer_tgt: Tokenizer, max_len: int, device):
    sos_idx = tokenizer_tgt.token_to_id(SOS)
    eos_idx = tokenizer_tgt.token_to_id(EOS)

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)

    # Initialize the decoder input with the sos token
    # two dimensions. One for batch one for the decoder input
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # Build mask for the target (decoder input)
        decoder_mask = casual_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculate the output of the decoder
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # Get the next token
        prob = model.project(out[:, -1])

        # Select the token with the max probability (because it is a greedy search)
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        # break if we predict the end of sentence token
        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


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

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

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

    for epoch in range(initial_epoch, config['num_epochs']):
        if (device == 'cuda'):
            torch.cuda.empty_cache()

        model.train()  # moved inside for run_validation at each step
        batch_iterator = tqdm(train_dataloader, desc=f'Processing epoch {epoch:02d}')
        for batch in batch_iterator:

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

            # Initialize to None instead of 0. Supposed to provide better performance.
            # optimizer.zero_grad()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        # Run validation at the end of each epoch
        validate_model1(model, val_dataloader, tokenizer_src, tokenizer_tgt,
                        config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)

        # Save the model at the end of every epoch
        save_model(model, optimizer, epoch, global_step)


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

    for epoch in range(initial_epoch, config['num_epochs']):
        if (device == 'cuda'):
            torch.cuda.empty_cache()

        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f'Processing epoch {epoch:02d}')
        for batch in batch_iterator:
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

            global_step += 1
            # print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

        # Save the model at the end of every epoch
        save_model(model, optimizer, epoch, global_step)


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

    for epoch in range(initial_epoch, config['num_epochs']):
        if (device == 'cuda'):
            torch.cuda.empty_cache()

        model.train()  # moved inside for run_validation at each step

        total_loss = 0

        batch_iterator = tqdm(train_dataloader, desc=f'Processing epoch {epoch:02d}')
        for batch in batch_iterator:

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

            total_loss += loss.item()

        # Save the model at the end of every epoch
        save_model(model, optimizer, epoch, global_step)


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

    tokenizer_src = None
    tokenizer_tgt = None
    train_dataloader, val_dataloader, src_vocab_size, tgt_vocab_size, src_to_index, tgt_to_index = get_ds6(
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

    for epoch in range(initial_epoch, config['num_epochs']):
        if (device == 'cuda'):
            torch.cuda.empty_cache()

        transformer.train()  # moved inside for run_validation at each step
        batch_iterator = tqdm(train_dataloader, desc=f'Processing epoch {epoch:02d}')
        for batch in batch_iterator:

            # ORG
            # for epoch in range(num_epochs):
            #     print(f"Epoch {epoch}")
            #     iterator = iter(train_loader)
            #     for batch_num, batch in enumerate(iterator):

            # eng_batch: tuple[str], kn_batch: tuple[str]
            eng_batch, kn_batch = batch
            encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask = Dataset6.create_masks(
                eng_batch, kn_batch, config['seq_len'])
            optimizer.zero_grad()
            kn_predictions = transformer(eng_batch,
                                         kn_batch,
                                         encoder_self_attention_mask.to(device),
                                         decoder_self_attention_mask.to(device),
                                         decoder_cross_attention_mask.to(device),
                                         enc_start_token=False,  # During training, model6 does not add sos to encoder input
                                         enc_end_token=False,  # During training, model6 does not add sos to encoder input
                                         dec_start_token=True,  # During training, model6 DOES add sos to decoder input
                                         dec_end_token=True)  # During training, model6 DOES add eos to decoder input
            labels = transformer.decoder.sentence_embedding.batch_tokenize(kn_batch, start_token=False, end_token=True)
            loss = loss_fn(kn_predictions.view(-1, tgt_vocab_size).to(device), labels.view(-1).to(device)).to(device)

            valid_indicies = torch.where(labels.view(-1) == tgt_to_index[PAD], False, True)
            loss = loss.sum() / valid_indicies.sum()
            batch_iterator.set_postfix({"Loss": f"{loss:6.3f}"})

            # Log of loss
            writer.add_scalar('train loss', loss, global_step)
            writer.flush()

            loss.backward()
            optimizer.step()

            # train_losses.append(loss.item())
            # if batch_num % 100 == 0:
            #     print(f"Iteration {batch_num} : {loss.item()}")
            #     print(f"English: {eng_batch[0]}")
            #     print(f"Kannada Translation: {kn_batch[0]}")
            #     kn_sentence_predicted = torch.argmax(kn_predictions[0], axis=1)
            #     predicted_sentence = ""
            #     for idx in kn_sentence_predicted:
            #         if idx == kannada_to_index[END_TOKEN]:
            #             break
            #         predicted_sentence += index_to_kannada[idx.item()]
            #     print(f"Kannada Prediction: {predicted_sentence}")

        # Run validation at the end of each epoch
        # validate_model6(transformer, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)

        # Save the model at the end of every epoch
        save_model(transformer, optimizer, epoch, global_step)


def validate_model6(transformer: Transformer6, validation_ds: DataLoader, src_to_index: dict, tgt_to_inddex: dict,
                    max_len: int, device, print_msg, global_step: int, writer, num_examples: int = 2):
    transformer.eval()
    count = 0

    if True:
        transformer.eval()
        kn_sentence: tuple[str] = ("",)
        eng_sentence: tuple[str] = ("should we go to the mall?",)
        for word_counter in range(max_len):
            encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask = Dataset6.create_masks(
                eng_sentence, kn_sentence)
            predictions = transformer(eng_sentence,
                                      kn_sentence,
                                      encoder_self_attention_mask.to(device),
                                      decoder_self_attention_mask.to(device),
                                      decoder_cross_attention_mask.to(device),
                                      enc_start_token=False,  # During validation, model6 does NOT add sos to encoder input
                                      enc_end_token=False,  # During validation, model6 does NOT add sos to encoder input
                                      dec_start_token=True,  # During validation, model6 DOES add sos to decoder input
                                      dec_end_token=False)  # During validation, model6 does NOT add eos to decoder input
            next_token_prob_distribution = predictions[0][word_counter]  # not actual probs
            next_token_index = torch.argmax(next_token_prob_distribution).item()
            next_token = index_to_kannada[next_token_index]
            kn_sentence = (kn_sentence[0] + next_token, )
            if next_token == EOS:
                break

        print(f"Evaluation translation (should we go to the mall?) : {kn_sentence}")
        print("-------------------------------------------")


if __name__ == '__main__':
    # warnings.filterwarnings('ignore')
    config = get_config()

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
        case _:
            train_model1(config)
