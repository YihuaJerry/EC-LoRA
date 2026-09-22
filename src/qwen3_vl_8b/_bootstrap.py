"""Expose this backbone's grouped source directories to legacy imports."""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent
for _path in (str(_ROOT.parent / "common"), *(str(_ROOT / name) for name in ("tools", "model", "eval"))):
    if _path not in sys.path:
        sys.path.insert(0, _path)
