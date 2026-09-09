# checkpoint-management Specification

## Purpose

Keep training resumable and checkpoints portable across machines.

Every model writes to a folder derived from its config rather than a hard-coded path, so a
run can be interrupted and resumed, and a whole run folder can be copied between an Apple
Silicon box and a CUDA one. The `.pt` carries weights only, which is why the folder — not
the file — is the unit that travels: the matching `config.yaml` (architecture dims) and
`tokenizer_*.json` (vocab size) have to arrive with it.

## Requirements

### Requirement: Deterministic checkpoint folder layout

Checkpoint paths SHALL be derived from configuration: the model folder is `<datasource>_<lang_src>_<lang_tgt>_<alt_model>` (or without the model suffix when `alt_model` is falsy), per-epoch weights are `<model_basename><epoch>.pt`, and the best-model file is `best_model_params.pt`.

#### Scenario: Weights path resolution

- **WHEN** `get_weights_file_path(config, "07")` is called
- **THEN** it returns `<model_folder>/<model_basename>07.pt`

#### Scenario: Latest checkpoint discovery

- **WHEN** `latest_weights_file_path(config)` is called and checkpoints exist
- **THEN** it returns the lexicographically last matching `<model_basename>*` file, else `None`

### Requirement: Save and restore training state

`save_model` SHALL persist epoch, model state, optimizer state, and global step. `reload_model` SHALL restore them for resuming training, and `load_trained_model` SHALL restore model weights for inference, raising if no checkpoint exists.

#### Scenario: Round-trip resume

- **WHEN** `save_model` writes a checkpoint and `reload_model` later loads it
- **THEN** the returned `initial_epoch` is the saved epoch + 1 and optimizer/global-step state is restored

#### Scenario: Best-model save uses correct arity

- **WHEN** `save_model(..., best_model_yet=True)` runs
- **THEN** the best-model path is resolved by calling `get_best_model_params_path(config)` with the arguments its signature accepts (no `TypeError` from a surplus argument)

#### Scenario: Inference load requires a checkpoint

- **WHEN** `load_trained_model(config, model)` is called and no checkpoint is resolvable
- **THEN** a `ValueError` is raised instead of returning an untrained model
