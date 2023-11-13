from pathlib import Path

import torch
import torch.nn as nn
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import (EOS, PAD, get_console_width, get_device, get_model_folder, get_config)
from dataset2 import get_ds2, get_testing_ds2
from model2 import Transformer2, build_transformer2
from utils import collect_training_metrics, reload_model, save_model, load_trained_model


def build_model2(config: dict, vocab_src_len: int, vocab_tgt_len: int) -> Transformer2:
    model = build_transformer2(vocab_src_len, vocab_tgt_len, config['seq_len'],
                               d_model=config['d_model'], N=config['N'], h=config['h'], dropout=config['dropout'], d_ff=config['d_ff'])
    return model


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
                batch_iterator.write(f"{'Source: ':>15}{batch['src_text'][0]}")
                batch_iterator.write(f"{'Target: ':>15}{batch['tgt_text'][0]}")
                kn_sentence_predicted = torch.argmax(output[0], axis=1)
                # JEB: Figure out how to get decode to stop at eos
                # predicted_sentence = tokenizer_tgt.decode(kn_sentence_predicted.detach().cpu().numpy(), skip_special_tokens=True)
                predicted_words = []
                for idx in kn_sentence_predicted:
                    if idx == tokenizer_tgt.token_to_id(EOS):
                        break
                    predicted_words.append(tokenizer_tgt.id_to_token(idx.item()))
                predicted_sentence = ' '.join(predicted_words)
                batch_iterator.write(f"{'Prediction: ':>15}{predicted_sentence}")
                batch_iterator.write('-' * console_width)

            global_step += 1
            # print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

        # Save the model at the end of every epoch
        save_model(config, model, optimizer, epoch, global_step)


def evaluate_model2(model: Transformer2, validation_ds: DataLoader, tokenizer_src: Tokenizer, tokenizer_tgt: Tokenizer,
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
            print_msg(f"{'SOURCE: ':>12}{source_text}")
            print_msg(f"{'TARGET: ':>12}{target_text}")
            print_msg(f"{'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-' * console_width)
                break
    if writer:
        collect_training_metrics(writer, predicted, expected, global_step)

def translate2(config: dict, sentence: str):
    device = get_device()

    model_folder = get_model_folder(config)
    if not Path.exists(Path(model_folder)):
        raise ValueError(f"{model_folder} model_folder does not exist")

    sentence, label, tokenizer_src, tokenizer_tgt = get_testing_ds2(config, model_folder, sentence)
    model = build_model2(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    # Load the pretrained weights
    model = load_trained_model(config, model)

    # if the sentence is a number use it as an index to the test set
    # run_translation(label, sentence, model, tokenizer_src, tokenizer_tgt, config['seq_len'], device)

def debug_code_model2(config: dict, device):
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

if __name__ == '__main__':
    # warnings.filterwarnings('ignore')
    config = get_config()
    device = get_device()
    debug_code_model2(config, device)
