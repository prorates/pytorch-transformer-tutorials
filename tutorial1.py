from pathlib import Path

# This is based on the following [video](https://youtu.be/ISNdQcPhsts)
# The code is original code is available [here](https://github.com/hkproj/pytorch-transformer)

import torch
import torch.nn as nn
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import EOS, PAD, SOS, get_console_width, get_device, get_model_folder, get_config
from dataset1 import get_ds1, get_testing_ds1
from model1 import Transformer1, build_transformer1
from utils import collect_training_metrics, reload_model, save_model, load_trained_model


def build_model1(config: dict, vocab_src_len: int, vocab_tgt_len: int) -> Transformer1:
    model = build_transformer1(
        vocab_src_len,
        vocab_tgt_len,
        config["seq_len"],
        config["seq_len"],
        d_model=config["d_model"],
        N=config["N"],
        h=config["h"],
        dropout=config["dropout"],
        d_ff=config["d_ff"],
    )
    return model


def evaluate_model1(
    model: Transformer1,
    validation_ds: DataLoader,
    tokenizer_src: Tokenizer,
    tokenizer_tgt: Tokenizer,
    max_len: int,
    device,
    print_msg,
    global_step: int,
    writer,
    num_examples: int = 2,
):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    console_width = get_console_width()

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)

            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            eos_idx = tokenizer_tgt.token_to_id(EOS)
            sos_idx = tokenizer_tgt.token_to_id(SOS)
            # JEB: Not sure this is really consistent
            # encoder_input has shape (bs=1, SeqLen)
            # encoder_mask has shape (bs=1, 1, 1, SeqLen)
            # model_out has shape (SeqLen)
            model_out = model.greedy_decode(encoder_input, encoder_mask, sos_idx, eos_idx, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            # Print the message to the console without interfering with the progress bar
            print_msg("-" * console_width)
            print_msg(f"{'Source: ':>15}{source_text}")
            print_msg(f"{'Target: ':>15}{target_text}")
            print_msg(f"{'Prediction: ':>15}{model_out_text}")

            if count == num_examples:
                print_msg("-" * console_width)
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
    writer = SummaryWriter(get_model_folder(config) + "/" + config["experiment_name"])

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)

    initial_epoch = 0
    global_step = 0
    model, initial_epoch, optimizer, global_step = reload_model(config, model, optimizer, initial_epoch, global_step)

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id(PAD), label_smoothing=0.1).to(device)

    console_width = get_console_width()

    for epoch in range(initial_epoch, config["num_epochs"]):
        if device == "cuda":
            torch.cuda.empty_cache()

        model.train()  # moved inside for run_validation at each step
        batch_iterator = tqdm(train_dataloader, desc=f"Processing epoch {epoch:02d}")
        for batch_num, batch in enumerate(batch_iterator):

            encoder_input = batch["encoder_input"].to(device)  # (B, SeqLen)
            decoder_input = batch["decoder_input"].to(device)  # (B, SeqLen)
            encoder_mask = batch["encoder_mask"].to(device)  # (B, 1, 1, SeqLen)
            decoder_mask = batch["decoder_mask"].to(device)  # (B, 1, SeqLen, SeqLen)

            # Run the tensors through the transformer
            encoder_output = model.encode(encoder_input, encoder_mask)  # (B, SeqLen, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)  # (B, SeqLen, d_model)
            proj_output = model.project(decoder_output)  # (B, SeqLen, tgt_vocab_size)

            # Compare the output with the label
            label = batch["label"].to(device)  # (B, SeqLen)

            # (B, SeqLen, tgt_vocab_size) --> (B * SeqLen, tgt_vocab_size)
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"Loss": f"{loss.item():6.3f}"})

            # Log of loss
            writer.add_scalar("train loss", loss.item(), global_step)
            writer.flush()

            # backpropagate the loss
            loss.backward()

            # update the weights
            optimizer.step()

            if (batch_num > 0) and (batch_num % 100 == 0):
                batch_iterator.write("-" * console_width)
                batch_iterator.write(f"{'Source: ':>15}{batch['src_text'][0]}")
                batch_iterator.write(f"{'Target: ':>15}{batch['tgt_text'][0]}")
                kn_sentence_predicted = torch.argmax(proj_output[0], axis=1)
                # JEB: Figure out how to get decode to stop at eos
                # predicted_sentence = tokenizer_tgt.decode(kn_sentence_predicted.detach().cpu().numpy(), skip_special_tokens=True)
                predicted_words = []
                for idx in kn_sentence_predicted:
                    if idx == tokenizer_tgt.token_to_id(EOS):
                        break
                    predicted_words.append(tokenizer_tgt.id_to_token(idx.item()))
                predicted_sentence = " ".join(predicted_words)
                batch_iterator.write(f"{'Prediction: ':>15}{predicted_sentence}")
                batch_iterator.write("-" * console_width)

            # Initialize to None instead of 0. Supposed to provide better performance.
            # optimizer.zero_grad()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        # Run validation at the end of each epoch
        evaluate_model1(
            model, val_dataloader, tokenizer_src, tokenizer_tgt, config["seq_len"], device, lambda msg: batch_iterator.write(msg), global_step, writer
        )

        # Save the model at the end of every epoch
        save_model(config, model, optimizer, epoch, global_step)


def translate1(config: dict, sentence: str):
    device = get_device()

    model_folder = get_model_folder(config)
    if not Path.exists(Path(model_folder)):
        raise ValueError(f"{model_folder} model_folder does not exist")

    sentence, label, tokenizer_src, tokenizer_tgt = get_testing_ds1(config, model_folder, sentence)
    model = build_model1(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    # Load the pretrained weights
    model = load_trained_model(config, model)

    # if the sentence is a number use it as an index to the test set
    sos_token = torch.tensor([tokenizer_tgt.token_to_id(SOS)], dtype=torch.int64)
    eos_token = torch.tensor([tokenizer_tgt.token_to_id(EOS)], dtype=torch.int64)
    pad_token = torch.tensor([tokenizer_tgt.token_to_id(PAD)], dtype=torch.int64)

    model.eval()
    with torch.no_grad():
        # Precompute the encoder output and reuse it for every generation step
        # source = tokenizer_src.encode(sentence)
        # source = torch.cat([
        #     torch.tensor([tokenizer_src.token_to_id(SOS)], dtype=torch.int64),
        #     torch.tensor(source.ids, dtype=torch.int64),
        #     torch.tensor([tokenizer_src.token_to_id(EOS)], dtype=torch.int64),
        #     torch.tensor([tokenizer_src.token_to_id(PAD)] * (max_len - len(source.ids) - 2), dtype=torch.int64)
        # ], dim=0).to(device)
        enc_input_tokens = tokenizer_src.encode(sentence).ids
        enc_num_padding_tokens = config["seq_len"] - len(enc_input_tokens) - 2
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

        eos_idx = tokenizer_tgt.token_to_id(EOS)
        sos_idx = tokenizer_tgt.token_to_id(SOS)

        # source shape is expected to be (bs=1, SeqLen)
        # source_mask shape is expected to be (bs=1, 1, 1, SeqLen)
        model_out = model.greedy_decode(source.unsqueeze(0), source_mask, eos_idx, sos_idx, config["seq_len"], device)
        model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

    # Print the source sentence and target start prompt
    print(f"{'Source: ':>15}{sentence}")
    if label != "":
        print(f"{'Target: ':>15}{label}")
    print(f"{'Prediction: ':>15}{model_out_text}")
    return model_out_text


def debug_code_model1(config: dict, device):
    config["model"] = "model1"
    config["datasource"] = "opus_books"
    config["lang_src"] = "en"
    config["lang_tgt"] = "it"

    model_folder = get_model_folder(config)
    Path(model_folder).mkdir(parents=True, exist_ok=True)

    # train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds1(config, model_folder)
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds1(config, model_folder)
    model = build_model1(config, 500, 500).to(device)

    print(model)
    model.train()


if __name__ == "__main__":
    # warnings.filterwarnings('ignore')
    config = get_config()
    device = get_device()
    debug_code_model1(config, device)
