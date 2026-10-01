import pytest

from decorrelated_ensemble.evaluation.artifacts import (
    finalize_manifest,
    verify_manifest,
    write_json,
)


def test_hash_manifest_detects_mutation(tmp_path):
    write_json(tmp_path / "artifact.json", {"score": 0.5})
    finalize_manifest(tmp_path)
    assert verify_manifest(tmp_path)
    (tmp_path / "artifact.json").write_text("changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_manifest(tmp_path)
