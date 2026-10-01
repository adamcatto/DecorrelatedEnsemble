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


def test_prediction_export_allows_only_missing_models(tmp_path):
    (tmp_path / "method").mkdir()
    (tmp_path / "method/model.joblib").write_bytes(b"model")
    write_json(tmp_path / "prediction.json", {"score": 0.5})
    finalize_manifest(tmp_path)
    (tmp_path / "method/model.joblib").unlink()
    assert verify_manifest(tmp_path, allow_missing_models=True)
    with pytest.raises(FileNotFoundError):
        verify_manifest(tmp_path)
    (tmp_path / "prediction.json").unlink()
    with pytest.raises(FileNotFoundError):
        verify_manifest(tmp_path, allow_missing_models=True)
