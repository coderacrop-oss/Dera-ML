import os
import stat

from atomic_io import atomic_write_text


def test_atomic_write_honours_umask_like_open(tmp_path):
    old = os.umask(0o022)
    try:
        target = tmp_path / "out.csv"
        atomic_write_text(str(target), "a,b\n")
        plain = tmp_path / "plain.csv"
        plain.write_text("a,b\n")
    finally:
        os.umask(old)
    assert stat.S_IMODE(target.stat().st_mode) == stat.S_IMODE(plain.stat().st_mode) == 0o644
