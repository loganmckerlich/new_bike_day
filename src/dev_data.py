"""Dev-mode data loader — returns static sample data from ``data/dev/`` JSON files.

No network calls are made; this module reads only files bundled with the repository.
The returned structures are identical to what :func:`src.fetch.ingest_all` produces so
the rest of the app works without any code changes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# Absolute path to the bundled sample data directory.
_DEV_DIR: Path = Path(__file__).resolve().parents[1] / "data" / "dev"


def load_dev_data() -> dict[str, pd.DataFrame | dict[str, str]]:
    """Load static sample data from ``data/dev/`` JSON files.

    Returns the same structure as :func:`src.fetch.ingest_all`:

    - ``"bikes"``: ``dict[str, str]`` — gear_id → bike name.
    - ``"segments"``: :class:`pandas.DataFrame` of starred segments.
    - ``"efforts"``: :class:`pandas.DataFrame` of segment efforts with
      ``gear_id`` already resolved.

    Raises:
        FileNotFoundError: If any of the expected JSON files are missing.
    """
    bikes: dict[str, str] = json.loads((_DEV_DIR / "bikes.json").read_text())
    segments = pd.DataFrame(json.loads((_DEV_DIR / "segments.json").read_text()))
    efforts = pd.DataFrame(json.loads((_DEV_DIR / "efforts.json").read_text()))
    return {"bikes": bikes, "segments": segments, "efforts": efforts}
