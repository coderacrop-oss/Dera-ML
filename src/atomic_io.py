"""Small I/O helpers shared by the scheduler and CRM export (stdlib only)."""
from __future__ import annotations

import os
import tempfile


def atomic_write_text(path: str, text: str) -> None:
    """Write `text` to `path` atomically: temp file in the same dir, then os.replace.

    Readers (e.g. the CRM API) never see a half-written file.
    """
    directory = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp-", suffix=".part")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
