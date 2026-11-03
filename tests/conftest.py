import pytest

import pipeline_logger


@pytest.fixture(autouse=True)
def _isolated_audit_log(tmp_path_factory, monkeypatch):
    """Tests must never append to the repo's real output/pipeline_audit.jsonl."""
    monkeypatch.setattr(pipeline_logger, "_AUDIT_FILE", tmp_path_factory.mktemp("audit") / "pipeline_audit.jsonl")
