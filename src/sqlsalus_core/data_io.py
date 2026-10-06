"""Read plain or losslessly compressed JSON benchmark inputs."""

import gzip
import json
from pathlib import Path


def load_json(path: Path):
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as handle:
        return json.load(handle)
