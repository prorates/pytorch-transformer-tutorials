# Pytorch Transformer Tutorials

**Eight independent transformer implementations** (`model1`–`model8`), each transcribed
from a different public tutorial, video, or blog, then adapted to run on new datasets and
brought up to a common `ruff` + `mypy` standard. The implementations are kept deliberately
parallel — the value is in *comparing* them, not unifying them.

Every model `N` is a triad of files behind a uniform contract: `modelN.py` (the network),
`datasetN.py` (data + tokenizer + masks), `tutorialN.py` (`train_modelN` / `translateN` /
`debug_code_modelN`). Three CLIs — `train.py`, `translate.py`, `test.py` — read a config,
then `match config["alt_model"]` to dispatch to the selected model. See
[`architecture.md`](architecture.md) for the full design.

## The eight implementations

| # | Source | Dataset (`datasource`) | Task | Trains? |
|---|--------|------------------------|------|---------|
| **1** | [hkproj/pytorch-transformer](https://github.com/hkproj/pytorch-transformer) ([video](https://youtu.be/ISNdQcPhsts)) | `opus_books` (HF `datasets`) | Translation en→it / en→fr | ✅ reference encoder-decoder |
| **2** | [Towards Data Science](https://towardsdatascience.com/build-your-own-transformer-from-scratch-using-pytorch-84c850470dcb) / [DataCamp](https://www.datacamp.com/tutorial/building-a-transformer-with-py-torch) | `translate` (local CSV) | Translation en→fr | ✅ |
| **3** | [SamLynnEvans/Transformer](https://github.com/SamLynnEvans/Transformer) | `translate` (local CSV) | Translation en→fr | ✅ |
| **4** | [Karpathy — build a GPT](https://youtu.be/kCc8FmEb1nY) | `translate` (local CSV) | (GPT) | ⚠️ training stub — raises |
| **5** | Official PyTorch `nn.Transformer` | `translate` (local CSV) | Translation en→fr | ⚠️ training stub — raises |
| **6** | [ajhalthor/Transformer-Neural-Network](https://github.com/ajhalthor/Transformer-Neural-Network) ([playlist](https://www.youtube.com/playlist?list=PLTl9hO2Oobd97qfWC40gOSU8C0iu0m2l4)) | `translate` (local `.txt`) | Translation en→fr / en→kn | ✅ |
| **7** | [PyTorch Transformer tutorial](https://pytorch.org/tutorials/beginner/transformer_tutorial.html) | WikiText-2 (HF `datasets`) | Language modeling | ✅ |
| **8** | [Karpathy nanoGPT](https://youtu.be/kCc8FmEb1nY) ([colab](https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing)) | `tinyshakespeare` | Char-level generation | ✅ **flagship demo** |

`model4` and `model5` build cleanly (model + dataloaders) but their training loops are
intentionally unimplemented — `train_model{4,5}` raise `RuntimeError`. They dispatch and
are inference-shaped, but won't train yet.

## Quick start

```bash
# 1. Environment (pick the file matching your hardware)
python3 -m venv .venv && source .venv/bin/activate     # or: uv venv .venv
pip install -r mps-requirements.txt                    # Apple Silicon
# pip install -r cuda-requirements.txt                 # NVIDIA / WSL — see section below

# 2. Train (the device is auto-selected: CUDA → MPS → CPU)
python train.py                                        # default: model8 on tinyshakespeare
TOKENIZERS_PARALLELISM=false python train.py -m opus_books_en_it_model1   # model1, en→it

# 3. Inference (requires a trained checkpoint)
python translate.py                                                       # generate Shakespeare (model8)
python translate.py -m opus_books_en_it_model1 -s "I am not a very good student."

# 4. Quick build/shape smoke check across models
python test.py
```

**CLI flags:** both `train.py` and `translate.py` take `-c <config.yaml>` or
`-m <model_folder>`; `translate.py` also takes `-s <sentence>`. With neither `-c` nor `-m`,
the config auto-generates `tinyshakespeare_en_en_model8/` from `get_default_config()`.
A model must be **trained before inference** — no checkpoints are committed to the repo.

## Datasets

The `datasource` config field selects how `datasetN.py` loads data:

- **`opus_books` / WikiText-2** — downloaded on first run via HuggingFace `datasets`
  (model1 translation, model7 language modeling). No manual setup.
- **`tinyshakespeare`** — the `input.txt` corpus for the model8 char-level demo.
- **`translate`** — *local* data under `custom_datasets/<datasource>_<src>_<tgt>/`
  (CSV `dataset.csv` for model2/3, `<lang>.txt` pairs for model6). These files are **not**
  committed, so models 2/3/4/5/6 need you to provide that directory before they run.

## Project layout

```
config.py        # get_config · get_device (CUDA→MPS→CPU) · checkpoint path helpers
utils.py         # save_model / reload_model / load_trained_model · metrics (CER/WER/BLEU)
train.py         # training CLI      ── match config["alt_model"] → train_modelN(config)
translate.py     # inference CLI     ── match config["alt_model"] → translateN(config, sentence)
test.py          # debug smoke driver ── debug_code_modelN(config, device)
model{1..8}.py   # the networks
dataset{1..8}.py # data loading, tokenizers, attention masks
tutorial{1..8}.py# per-model train / translate / debug orchestration
<datasource>_<src>_<tgt>_<model>/   # per-run folder: config.yaml + tmodel_<epoch>.pt checkpoints
```

See [`architecture.md`](architecture.md) for the dispatch flow, checkpoint lifecycle, and
where to make changes.

## Running on CUDA / WSL

The code is device-agnostic: `get_device()` (`config.py`) auto-selects `cuda` when an
NVIDIA GPU is available, `mps` on Apple Silicon, otherwise `cpu`. The same commands work
everywhere — only the environment setup differs.

### 1. WSL2 + CUDA driver (one-time, Windows side)

CUDA-on-WSL exposes the **Windows** GPU driver into the Linux guest. Do **not** install a
Linux NVIDIA driver inside WSL — that breaks the passthrough.

- On Windows: install the latest NVIDIA driver (Game Ready / Studio already ships WSL CUDA support).
- Use WSL2 (`wsl --set-default-version 2`) with Ubuntu 22.04/24.04.
- Confirm the GPU is visible inside WSL: `nvidia-smi` should list your card.

### 2. Python environment (inside WSL)

```bash
python3 -m venv .venv            # or: uv venv .venv
source .venv/bin/activate
pip install -r cuda-requirements.txt
```

The CUDA wheels are prebuilt — you normally need **no C++ toolchain**. If pip falls back to a
source build (usually a Python-version mismatch), install build tools with
`sudo apt install build-essential python3-dev`, or switch to Python 3.11/3.12 which have wheels.

Verify torch sees the GPU:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 3. Train / translate (identical commands on any backend)

```bash
TOKENIZERS_PARALLELISM=false python train.py -m opus_books_en_it_model1   # model1, en->it
python train.py                                                          # default model8, tinyshakespeare
python translate.py -m opus_books_en_it_model1                           # inference
```

`train.py` takes `-c <config.yaml>` or `-m <modelfolder>`; with neither it auto-generates
`tinyshakespeare_en_en_model8/`. On startup the device line confirms the backend
(`Using NVIDIA GPU and device cuda` + card name/memory).

### Notes on limited-VRAM cards

- On `CUDA out of memory`, lower `batch_size` in the model folder's `config.yaml` (e.g. 8 → 4 → 2).
  Attention memory is O(`seq_len`²), so `seq_len` is the other lever, but it changes model capacity.
- Checkpoints (`tmodel_*.pt`) are portable across CUDA/MPS, but the committed configs were tuned for
  a 24 GB Apple Silicon box — on a roomier CUDA card you can raise `batch_size` for more throughput.
- Older cards without tensor cores (e.g. GTX 16-series) run fp32 fine; mixed precision won't help much.
- `cuda-requirements.txt` (torch 2.3.1) has drifted from `mps-requirements.txt` (torch 2.5.1); bump the
  pin to 2.5.1 if you hit CUDA op issues.

## Tested environments

Measured on the `model1` opus_books run (`-m opus_books_en_it_model1`: d_model=256, N=3,
seq_len=350) at the committed `batch_size=8`:

| Hardware | Backend | Python | it/s (model1, batch 8) |
|----------|---------|--------|------------------------|
| Mac mini M4 Pro | mps | 3.12 | ~8.5 |
| GTX 1660 (6 GB), WSL2 Ubuntu 26.04, CUDA 13.0 | cuda | 3.12 | ~6.5 |

Notes:

- **Python:** there are no PyTorch wheels for 3.14 yet, so use **3.12** (prebuilt `cp312`
  wheels exist for both the CUDA 2.3.1 and MPS 2.5.1 pins). On 3.14 pip falls back to a
  source build and needs a C++ toolchain.
- **System CUDA version doesn't need to match the wheel.** The torch wheel bundles its own
  CUDA 12.x runtime; a CUDA 13.0 *driver* only needs to be ≥ the wheel's runtime (it's
  backward-compatible). You don't need a matching system CUDA toolkit.
- **Small-batch caveat:** at `batch_size=8` the workload is bandwidth-/latency-bound, which
  flatters the M4's unified memory. A discrete GPU with VRAM headroom (the 1660 uses only
  ~2.6/6 GB here) amortizes launch overhead better at larger batches — raising `batch_size`
  typically improves its it/s.

### Full-run benchmark (model1, opus_books en→it)

First end-to-end reference run, to anchor comparative benchmarks. Config: d_model=256, N=3,
h=4, seq_len=350, batch_size=8, 23 epochs, 3638 steps/epoch.

| Hardware | Backend | it/s | sec/epoch | Total (23 ep) | Final loss |
|----------|---------|------|-----------|---------------|------------|
| Mac mini M4 Pro | mps | ~8.5 | ~427 | ~2h45m | ~2.8 (ep22) |
| GTX 1660 (6 GB), WSL2 | cuda | ~6.5 | _tbd_ | _tbd_ | _tbd_ |

Loss trajectory (M4, end-of-epoch): 10.0 → 6.0 (ep0) → ~3.5 (ep14) → ~2.8 (ep20–22).
_CUDA-side numbers fill in as they're measured._

**Inference sanity check** (`translate.py -m opus_books_en_it_model1`, on `tmodel_22.pt`):

```
Source: She opened the door and looked at the garden.
Pred:   Ella aprì la porta e guardò il giardino.       # ✅ essentially correct
Source: I am not a very good student.
Pred:   Io non sono stato messo in persona di privato.  # rougher — 23 epochs is light
```

## Checkpoint portability (macOS arm64 ↔ WSL amd64)

PyTorch `.pt` checkpoints are **architecture- and backend-portable** — a model trained on a
Mac (MPS) loads on a CUDA/WSL box and vice versa. **No ONNX conversion is needed** for
PyTorch-to-PyTorch use; ONNX is only for running outside PyTorch. `save_model` stores a
plain state-dict (tensors only, no pickled class), and `reload_model` / `load_trained_model`
pass `map_location=get_device()` so tensors are remapped onto the loading machine's backend.

To move a model between machines, copy the **entire run folder** — the `.pt` carries weights
only:

```
opus_books_en_it_model1/
├── config.yaml        # architecture dims — must match
├── tokenizer_*.json   # vocab size — must match (don't rebuild)
└── tmodel_*.pt        # weights
```

A `config.yaml` or tokenizer mismatch causes a `load_state_dict` shape error.

## Tutorial sources

Each model's original source is linked in [the table above](#the-eight-implementations).
SamLynnEvans' code was originally found via the FloydHub blog, and the official PyTorch
`nn.Transformer` (model5) was modified for tutorial purposes.

## Appendix: estimating memory requirements

> Reference notes on transformer memory math, adapted from
> [this blog post](https://schartz.github.io/blog/estimating-memory-requirements-of-transformers/).

`total_memory = memory_modal + memory_activations + memory_gradients`

### Estimating model's memory requirements
Lets take GPT as an example. GPT consists of a number of transformer blocks (let's call it n_tr_blocks from now on). Each transformer block consists of following structure:

```
multi_headed_attention --> layer_normalization --> MLP -->layer_normalization
```

Each multi_headed_attention element consists of value nets, key and query. Let's say that each of these have n_head attention heads and dim dimensions. MLP also has a dimension of n_head * dim. The memory needed to store these will be

```
total_memory = memory of multi_headed_attention + memory of MLP
			 = memory of value nets + memory of key + memory of query + memory of MLP
			 = square_of(n_head * dim) + square_of(n_head * dim) + square_of(n_head * dim) + square_of(n_head * dim)
			 = 4*square_of(n_head * dim)
```
Since our modal contains n_tr_blocks units of these blocks. Total memory required by the modal becomes.

```
memory_modal = 4*n_tr_blocks*square_of(n_head * dim)
```

Above estimation does not take into account the memory required for biases, since that is mostly static and does not depend on things like batch size, input sequence etc.


### Estimating model activation's memory requirements

Multi headed attention is generally a softmax. More specifically it can written as:

```
multi_headed_attention = softmax(query * key * sequence_length) * value_net
```
query key and value_net all have a tensor shape of

```
[batch_size, n_head, sequence_length, dim]
```

query * key * sequence_length operation gives following resultant shape:

```
[batch_size, n_head, sequence_length, sequence_length]
```
This finally gives the memory cost of activation function as

```
memory_softmax  = batch_size * n_head * square_of(sequence_length)
```


query * key * sequence_length operation multiplied by value_net has the shape of [batch_size, n_head, sequence_length, dim]. MLP also has the same shape. So memory cost of these operations become:

```
memory of MLP  = batch_size * n_head * sequence_length * dim
memory of value_net  = batch_size * n_head * sequence_length * dim
```

This gives us the memory of model activation per block:

mem_act = memory_softmax + memory_value_net + memory_MLP
		= batch_size * n_head * square_of(sequence_length)
		  + batch_size * n_head * sequence_length * dim
		  + batch_size * n_head * sequence_length * dim
		= batch_size * n_head * sequence_length * (sequence_length + 2*dim)
Memory of model activation across the model will be:

```
n_tr_blocks * (batch_size * n_head * sequence_length * (sequence_length + 2*dim))
````

### Summing it all up
To sum up total memory needed for fine-tuning/training transformer models is:

```
total_memory = memory_modal + 2 * memory_activations
```
Memory for modal is:

```
memory_modal = 4*n_tr_blocks*square_of(n_head * dim)
```
And memory for model activations is:

```
n_tr_blocks * (batch_size * n_head * sequence_length * (sequence_length + 2*dim))
```
These rough formulas can be written more succintly using following notation.

```
R = n_tr_blocks = number of transformer blocks in the model
N = n_head = number of attention heads
D = dim = dimension of each attention head
B = batch_size = batch size
S = sequence_length = input sequence length

memory modal = 4 * R * N^2 * D^2

memory activations = RBNS(S + 2D)
```
Total memory consumption if modal training is

```
M = (4 * R * N^2 * D^2) + RBNS(S + 2D)
```
If we have a very long sequence lengths S >> D S + 2D <--> S hence M in this case becomes:

```
M = (4 * R * N^2 * D^2) + RBNS(S) = 4*R*N^2*D^2 + RBNS^2

M is directly proportional to square of length of input sequence for large sequences
M is lineraly proportional to the batch size.
```

### Summary

These rough formula for estimating the memory requirements of fine tuning transformer models

```
R = n_tr_blocks = number of transformer blocks in the model
N = n_head = number of attention heads
D = dim = dimension of each attention head
B = batch_size = batch size
S = sequence_length = input sequence length


memory modal = 4 * R * N^2 * D^2

memory activations = RBNS(S + 2D)

total memory required = ((4 * R * N^2 * D^2) + RBNS(S + 2D)) * float64 memory in bytes
```

Insights
Memory consumption is directly proportional to square of length of input sequence for large sequences

Memory consumption is lineraly proportional to the batch size.

