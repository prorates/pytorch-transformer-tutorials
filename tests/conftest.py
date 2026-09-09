"""Put the repo root on sys.path.

The tutorial modules sit at the repo root and are imported by name, not from an
installed distribution (`[tool.uv] package = false`), so pytest's rootdir is not
automatically importable from inside tests/.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
