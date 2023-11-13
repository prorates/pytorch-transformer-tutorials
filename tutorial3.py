# The source code seems to be [here](https://github.com/SamLynnEvans/Transformer?ref=blog.floydhub.com)

from pathlib import Path

import torch
import torch.nn as nn
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import (EOS, PAD, get_console_width, get_device, get_model_folder, get_config)
from dataset3 import get_ds3, get_testing_ds3
from model3 import Transformer3, build_transformer3
from utils import collect_training_metrics, reload_model, save_model, load_trained_model


class CosineWithRestarts(torch.optim.lr_scheduler._LRScheduler):
    """
    Cosine annealing with restarts.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer

    T_max : int
        The maximum number of iterations within the first cycle.

    eta_min : float, optional (default: 0)
        The minimum learning rate.

    last_epoch : int, optional (default: -1)
        The index of the last epoch.

    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 T_max: int,
                 eta_min: float = 0.,
                 last_epoch: int = -1,
                 factor: float = 1.) -> None:
        # pylint: disable=invalid-name
        self.T_max = T_max
        self.eta_min = eta_min
        self.factor = factor
        self._last_restart: int = 0
        self._cycle_counter: int = 0
        self._cycle_factor: float = 1.
        self._updated_cycle_len: int = T_max
        self._initialized: bool = False
        super(CosineWithRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """Get updated learning rate."""
        # HACK: We need to check if this is the first time get_lr() was called, since
        # we want to start with step = 0, but _LRScheduler calls get_lr with
        # last_epoch + 1 when initialized.
        if not self._initialized:
            self._initialized = True
            return self.base_lrs

        step = self.last_epoch + 1
        self._cycle_counter = step - self._last_restart

        lrs = [
            (
                self.eta_min + ((lr - self.eta_min) / 2) *
                (
                    np.cos(
                        np.pi *
                        ((self._cycle_counter) % self._updated_cycle_len) /
                        self._updated_cycle_len
                    ) + 1
                )
            ) for lr in self.base_lrs
        ]

        if self._cycle_counter % self._updated_cycle_len == 0:
            # Adjust the cycle length.
            self._cycle_factor *= self.factor
            self._cycle_counter = 0
            self._updated_cycle_len = int(self._cycle_factor * self.T_max)
            self._last_restart = step

        return lrs


def build_model3(config: dict, vocab_src_len: int, vocab_tgt_len: int) -> Transformer3:
    model = build_transformer3(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'],
                               d_model=config['d_model'], n_layers=config['N'], heads=config['h'], dropout=config['dropout'])
    return model


def evaluate_model3(model: Transformer3, validation_ds: DataLoader, tokenizer_src: Tokenizer, tokenizer_tgt: Tokenizer,
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


def train_model3(config: dict):
    device = get_device()

    model_folder = get_model_folder(config)
    Path(model_folder).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds3(config, model_folder)
    model = build_model3(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], betas=(0.9, 0.98), eps=1e-9)
    if False:
        scheduler = CosineWithRestarts(opt.optimizer, T_max=opt.train_len)

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
                batch_iterator.write(f"{'Source: ':>15}{batch['src_text'][0]}")
                batch_iterator.write(f"{'Target: ':>15}{batch['tgt_text'][0]}")
                kn_sentence_predicted = torch.argmax(preds[0], axis=1)
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

            total_loss += loss.item()

        # Save the model at the end of every epoch
        save_model(config, model, optimizer, epoch, global_step)


def translate3(config: dict, sentence: str):
    device = get_device()

    model_folder = get_model_folder(config)
    if not Path.exists(Path(model_folder)):
        raise ValueError(f"{model_folder} model_folder does not exist")

    sentence, label, tokenizer_src, tokenizer_tgt = get_testing_ds3(config, model_folder, sentence)
    model = build_model3(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    # Load the pretrained weights
    model = load_trained_model(config, model)

    # if the sentence is a number use it as an index to the test set
    # run_translation(label, sentence, model, tokenizer_src, tokenizer_tgt, config['seq_len'], device)


def debug_code_model3(config: dict, device):
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


if __name__ == '__main__':
    # warnings.filterwarnings('ignore')
    config = get_config()
    device = get_device()
    debug_code_model3(config, device)
