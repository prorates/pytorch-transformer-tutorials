## ADDED Requirements

### Requirement: Runtime device selection

The system SHALL select a compute device at runtime in priority order CUDA → MPS → CPU, exposed through a single helper (`config.get_device()`), and every training and inference path SHALL obtain its device from that helper rather than hard-coding a backend. The selected device SHALL be reported to the user.

#### Scenario: CUDA available

- **WHEN** `torch.cuda.is_available()` is true
- **THEN** `get_device()` returns `"cuda"` and prints the device name and total memory

#### Scenario: Apple Silicon available

- **WHEN** CUDA is unavailable but `torch.backends.mps.is_available()` (or `is_built()`) is true
- **THEN** `get_device()` returns `"mps"` and reports that Apple Silicon is in use

#### Scenario: CPU fallback

- **WHEN** neither CUDA nor MPS is available
- **THEN** `get_device()` returns `"cpu"` and the workflow still runs to completion

### Requirement: Device-conditional operations are guarded

Operations that exist only on a specific backend (for example `torch.cuda.empty_cache()`) SHALL be guarded by a device check so the code does not raise on MPS or CPU.

#### Scenario: No CUDA-only call on non-CUDA device

- **WHEN** training runs on `mps` or `cpu`
- **THEN** no CUDA-only API is invoked and the epoch loop completes without raising
