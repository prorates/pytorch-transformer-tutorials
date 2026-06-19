## ADDED Requirements

### Requirement: Train a selected model from configuration

The `train.py` entrypoint SHALL load configuration via `config.get_config()` and dispatch to the trainer matching `config["alt_model"]` (`model1`–`model8`), defaulting to `model1` for an unrecognized value. Every model referenced by the dispatch table SHALL be importable so that selecting it does not raise `NameError`.

#### Scenario: Dispatch to the configured model

- **WHEN** `train.py` runs with `config["alt_model"] == "model8"`
- **THEN** `train_model8(config)` is invoked

#### Scenario: Every dispatch target is imported

- **WHEN** `train.py` is imported
- **THEN** every `train_model{1..8}` referenced in the `match` is bound (no commented-out import leaves a called name undefined)

#### Scenario: CLI overrides

- **WHEN** `train.py -c <config_file>` or `-m <model_folder>` is provided
- **THEN** configuration is loaded from that file/folder instead of the generated default

### Requirement: Checkpoint written each epoch

Training SHALL persist a checkpoint at the end of every epoch into the model folder resolved by `config.get_model_folder()`, capturing at minimum epoch, model state, optimizer state, and global step.

#### Scenario: End-of-epoch save

- **WHEN** an epoch of `train_model8` completes
- **THEN** a weights file `tmodel_<epoch>.pt` exists under `<datasource>_<src>_<tgt>_<model>/`

#### Scenario: Resume from latest checkpoint

- **WHEN** `config["preload"] == "latest"` and a prior checkpoint exists
- **THEN** training resumes from the saved epoch, optimizer state, and global step rather than from scratch
