"""Small I/O helpers shared by the scheduler, CRM export and model registry (stdlib only)."""
from __future__ import annotations

import os
import tempfile


def atomic_write_bytes(path: str, data: bytes) -> None:
    """Write `data` to `path` atomically: temp file in the same dir, then os.replace.

    Readers (e.g. the CRM API, a serving run) never see a half-written file.
    """
    directory = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp-", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        # mkstemp creates 0600; give the file the mode a plain open() would (0666 & ~umask) so a
        # reader running as another user (e.g. the CRM API service) can still read it.
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(tmp, 0o666 & ~umask)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def atomic_write_text(path: str, text: str) -> None:
    """Text version of `atomic_write_bytes` (UTF-8, no newline translation)."""
    atomic_write_bytes(path, text.encode("utf-8"))
