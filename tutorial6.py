# This is based on the following [video](https://www.youtube.com/playlist?list=PLTl9hO2Oobd97qfWC40gOSU8C0iu0m2l4)
# The code is original code is available [here](https://github.com/ajhalthor/Transformer-Neural-Network)

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import EOS, PAD, get_console_width, get_device, get_model_folder, get_config
from dataset6 import Dataset6, get_ds6, get_testing_ds6
from model6 import Transformer6, build_transformer6
from utils import collect_training_metrics, reload_model, save_model, load_trained_model


def build_model6(config: dict, vocab_src_len: int, vocab_tgt_len: int, src_to_index: dict, tgt_to_index: dict) -> Transformer6:
    model = build_transformer6(
        vocab_src_len,
        vocab_tgt_len,
        src_to_index,
        tgt_to_index,
        config["seq_len"],
        config["seq_len"],
        d_model=config["d_model"],
        N=config["N"],
        h=config["h"],
        dropout=config["dropout"],
        d_ff=config["d_ff"],
    )

    return model


def train_model6(config: dict):
    device = get_device()

    model_folder = get_model_folder(config)
    Path(model_folder).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, src_vocab_size, tgt_vocab_size, src_to_index, tgt_to_index, index_to_tgt = get_ds6(config, model_folder)
    transformer = build_model6(config, src_vocab_size, tgt_vocab_size, src_to_index, tgt_to_index).to(device)

    # Tensorboard
    writer = SummaryWriter(get_model_folder(config) + "/" + config["experiment_name"])

    optimizer = torch.optim.Adam(transformer.parameters(), lr=config["lr"])

    total_loss = 0
    initial_epoch = 0
    global_step = 0

    transformer, initial_epoch, optimizer, global_step = reload_model(config, transformer, optimizer, initial_epoch, global_step)
    # loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id(PAD), reduction='none')
    loss_fn = nn.CrossEntropyLoss(ignore_index=tgt_to_index[PAD], reduction="none")

    console_width = get_console_width()

    for epoch in range(initial_epoch, config["num_epochs"]):
        if device == "cuda":
            torch.cuda.empty_cache()

        transformer.train()  # moved inside for run_validation at each step
        batch_iterator = tqdm(train_dataloader, desc=f"Processing epoch {epoch:02d}")
        for batch_num, batch in enumerate(batch_iterator):

            # src_batched_sentences: tuple[str], tgt_batched_sentences: tuple[str]
            src_batched_sentences, tgt_batched_sentences = batch
            encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask = Dataset6.create_masks(
                src_batched_sentences, tgt_batched_sentences, config["seq_len"]
            )
            optimizer.zero_grad()
            predicted_tokens = transformer(
                src_batched_sentences,
                tgt_batched_sentences,
                encoder_self_attention_mask.to(device),
                decoder_self_attention_mask.to(device),
                decoder_cross_attention_mask.to(device),
                enc_start_token=False,  # During training, model6 does not add sos to encoder input
                enc_end_token=False,  # During training, model6 does not add sos to encoder input
                dec_start_token=True,  # During training, model6 DOES add sos to decoder input
                dec_end_token=True,
            )  # During training, model6 DOES add eos to decoder input
            expected_tokens = transformer.decoder.sentence_embedding.batch_tokenize(tgt_batched_sentences, start_token=False, end_token=True)
            loss = loss_fn(predicted_tokens.view(-1, tgt_vocab_size).to(device), expected_tokens.view(-1).to(device)).to(device)

            valid_indicies = torch.where(expected_tokens.view(-1) == tgt_to_index[PAD], False, True)
            loss = loss.sum() / valid_indicies.sum()
            batch_iterator.set_postfix({"Loss": f"{loss.item():6.3f}"})

            # Log of loss
            writer.add_scalar("train loss", loss, global_step)
            writer.flush()

            loss.backward()
            optimizer.step()

            # train_losses.append(loss.item())
            if (batch_num > 0) and (batch_num % 100 == 0):
                batch_iterator.write("-" * console_width)
                batch_iterator.write(f"{'Source: ':>15}{src_batched_sentences[0]}")
                batch_iterator.write(f"{'Target: ':>15}{tgt_batched_sentences[0]}")
                kn_sentence_predicted = torch.argmax(predicted_tokens[0], axis=1)
                predicted_sentence = ""
                for idx in kn_sentence_predicted:
                    if idx == tgt_to_index[EOS]:
                        break
                    predicted_sentence += index_to_tgt[idx.item()]
                batch_iterator.write(f"{'Prediction: ':>15}{predicted_sentence}")
                batch_iterator.write("-" * console_width)

            # if batch_num % 20 == 0:
            #     evaluate_model6(transformer, val_dataloader, index_to_tgt,
            #                     config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)

        # Run validation at the end of each epoch
        evaluate_model6(transformer, val_dataloader, index_to_tgt, config["seq_len"], device, lambda msg: batch_iterator.write(msg), global_step, writer)

        # Save the model at the end of every epoch
        save_model(config, transformer, optimizer, epoch, global_step)


def evaluate_model6(
    transformer: Transformer6, validation_ds: DataLoader, index_to_tgt: dict, max_len: int, device, print_msg, global_step: int, writer, num_examples: int = 2
):

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

            predicated_batched_sentences = transformer.greedy_decode(src_batched_sentences, max_len, index_to_tgt, device)

            source_texts.append(src_batched_sentences[0])
            expected.append(expected_batched_sentences[0])
            predicted.append(predicated_batched_sentences[0])

            # Print the message to the console without interfering with the progress bar
            print_msg("-" * console_width)
            print_msg(f"{'Source: ':>15}{src_batched_sentences[0]}")
            print_msg(f"{'Target: ':>15}{expected_batched_sentences[0]}")
            print_msg(f"{'Prediction: ':>15}{predicated_batched_sentences[0]}")

            if count == num_examples:
                print_msg("-" * console_width)
                break
    if writer:
        collect_training_metrics(writer, predicted, expected, global_step)


def translate6(config: dict, sentence: str):
    device = get_device()

    model_folder = get_model_folder(config)
    if not Path.exists(Path(model_folder)):
        raise ValueError(f"{model_folder} model_folder does not exist")

    sentence, label, vocab_src_len, vocab_tgt_len, src_to_index, tgt_to_index, index_to_tgt = get_testing_ds6(config, model_folder, sentence)
    model = build_model6(config, vocab_src_len, vocab_tgt_len, src_to_index, tgt_to_index).to(device)

    # Load the pretrained weights
    model = load_trained_model(config, model)

    # if the sentence is a number use it as an index to the test set
    # run_translation(label, sentence, model, tokenizer_src, tokenizer_tgt, config['seq_len'], device)A
    model.eval()
    with torch.no_grad():
        src_batched_sentences = (sentence.lower(),)
        predicated_batched_sentences = model.greedy_decode(src_batched_sentences, config["seq_len"], index_to_tgt, device)
        output_text = predicated_batched_sentences[0]

    # Print the source sentence and target start prompt
    print(f"{'Source: ':>15}{sentence}")
    if label != "":
        print(f"{'Target: ':>15}{label}")
    print(f"{'Prediction: ':>15}{output_text}")
    return output_text


def debug_code_model6(config: dict, device):
    config["model"] = "model6"
    config["datasource"] = "translate"
    config["lang_src"] = "en"
    config["lang_tgt"] = "kn"

    model_folder = get_model_folder(config)
    Path(model_folder).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, src_vocab_size, tgt_vocab_size, src_to_index, tgt_to_index, index_to_tgt = get_ds6(config, model_folder)
    model = build_model6(config, src_vocab_size, tgt_vocab_size, src_to_index, tgt_to_index).to(device)

    print(model)
    model.train()


if __name__ == "__main__":
    # warnings.filterwarnings('ignore')
    config = get_config()
    device = get_device()
    debug_code_model6(config, device)
