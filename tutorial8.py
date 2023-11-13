# See [video](https://youtu.be/kCc8FmEb1nY)
# The colab repo is [here](https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing)

import time
from pathlib import Path

import torch
from tqdm import tqdm

from config import get_config, get_device, get_model_folder
from dataset8 import get_ds8
from model8 import Transformer8, build_transformer8
from utils import reload_model, save_model


def build_model8(config: dict, vocab_tgt_len: int) -> Transformer8:
    model = build_transformer8(vocab_tgt_len,
                               d_model=config['d_model'], N=config['N'], h=config['h'], block_size=32, dropout=config['dropout'], d_ff=config['d_ff'])
    return model


def train_model8(config: dict):
    # hyperparameters
    max_iters = 5000
    eval_interval = 100
    eval_iters = 200
    total_loss = 0
    initial_epoch = 0
    global_step = 0

    torch.manual_seed(1337)

    device = get_device()

    model_folder = get_model_folder(config)
    Path(model_folder).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_tgt = get_ds8(config, model_folder)
    transformer = build_model8(config, tokenizer_tgt.get_vocab_size()).to(device)

    # print the number of parameters in the model
    print(sum(p.numel() for p in transformer.parameters())/1e6, 'M parameters')

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(transformer.parameters(), lr=config['lr'])

    transformer, initial_epoch, optimizer, global_step = reload_model(
        config, transformer, optimizer, initial_epoch, global_step)

    for epoch in range(initial_epoch, config['num_epochs']):
        if (device == 'cuda'):
            torch.cuda.empty_cache()

        transformer.train()  # moved inside for run_validation at each step

        batch_iterator = tqdm(train_dataloader, desc=f'Processing epoch {epoch:02d}')
        for iter, batch in enumerate(batch_iterator):
            if (iter == max_iters):
                break

            # every once in a while evaluate the loss on train and val sets
            if (iter % eval_interval == 0 or iter == max_iters - 1) and (iter > 0):
                losses = evaluate_model8(transformer, val_dataloader, eval_iters)
                batch_iterator.write(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            # sample a batch of data
            xb, yb = batch

            # evaluate the loss
            logits, loss = transformer(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        # Save the model at the end of every epoch
        save_model(config, transformer, optimizer, epoch, global_step)


    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(tokenizer_tgt.decode(transformer.generate(context, max_new_tokens=2000)[0].tolist()))


@torch.no_grad()
def evaluate_model8(transformer: Transformer8, validation_ds, eval_iters):
    out = {'train':0, 'val': 0}
    transformer.eval()
    # for split in ['train', 'val']:
    #     losses = torch.zeros(eval_iters)
    #     for k in range(eval_iters):
    #         X, Y = get_batch(split)
    #         logits, loss = transformer(X, Y)
    #         losses[k] = loss.item()
    #     out[split] = losses.mean()
    losses = torch.zeros(eval_iters)
    # for k in range(eval_iters):
    for k, batch in enumerate(validation_ds):
        if k == eval_iters:
            break
        X, Y = batch
        logits, loss = transformer(X, Y)
        losses[k] = loss.item()
    out['val'] = losses.mean()
    transformer.train()
    return out

def translate8(config: dict, sentence: str):
    device = get_device()

    model_folder = get_model_folder(config)
    if not Path.exists(Path(model_folder)):
        raise ValueError(f"{model_folder} model_folder does not exist")

    raise RuntimeError("Not implemented yet")

def debug_code_model8(config: dict, device):
    config['model'] = "model7"
    config['datasource'] = "translate"
    config['lang_src'] = "en"
    config['lang_tgt'] = "fr"

    model_folder = get_model_folder(config)
    Path(model_folder).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, test_dataloader, tokenizer_tgt = get_ds8(config, model_folder)
    model = build_model8(config, tokenizer_tgt.get_vocab_size()).to(device)

    print(model)
    model.train()


if __name__ == '__main__':
    # warnings.filterwarnings('ignore')
    config = get_config()
    device = get_device()
    debug_code_model8(config, device)
