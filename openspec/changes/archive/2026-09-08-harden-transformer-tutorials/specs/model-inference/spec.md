## ADDED Requirements

### Requirement: Infer from a trained checkpoint

The `translate.py` entrypoint SHALL load configuration, resolve the device via `config.get_device()`, dispatch to the handler matching `config["alt_model"]`, and produce output from a previously trained checkpoint. If the model folder or checkpoint is missing, the handler SHALL raise a clear error rather than silently producing garbage.

#### Scenario: Dispatch to the configured model

- **WHEN** `translate.py -s "<sentence>"` runs with `config["alt_model"] == "model2"`
- **THEN** `translate2(config, sentence)` is invoked

#### Scenario: Missing checkpoint is reported

- **WHEN** inference is requested but the resolved model folder or weights file does not exist
- **THEN** a `ValueError` is raised naming the missing path

### Requirement: Shakespeare-style generation for the GPT model

For `model8` (tinyshakespeare, character-level GPT), inference SHALL autoregressively generate text from a seed context and decode it with the model's tokenizer, producing Shakespeare-style output.

#### Scenario: Generate from empty context

- **WHEN** `translate8(config, sentence)` runs against a trained `model8` checkpoint
- **THEN** the model generates new tokens from a zero seed context and prints the decoded character stream
