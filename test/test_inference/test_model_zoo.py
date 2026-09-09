from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from puresound.inference import ModelZoo, ModelZooError, ModelZooValidationError


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_checked_in_catalog_has_expected_inventory_and_defaults():
    zoo = ModelZoo.default()
    assert len(zoo.catalog.models) == 10
    assert sum(len(model.artifacts) for model in zoo.catalog.models) == 11
    assert zoo.get("voice-isolate-dpcrn-v8").roles == ["default"]
    assert zoo.get("speaker-verification-ps-spk-v1-1").roles == ["default"]
    assert zoo.list(task="ns") == []
    assert zoo.list(task="tse") == []


def test_checked_in_catalog_validates_paths_hashes_sidecars_and_graphs():
    report = ModelZoo.default().validate()
    assert report.ok
    assert report.models == 10
    assert report.artifacts == 11


def test_duplicate_ids_are_rejected(tmp_path):
    catalog = {
        "schema_version": 1,
        "models": [
            {"id": "same", "display_name": "a", "task": "voice_isolation", "lifecycle": "released"},
            {"id": "same", "display_name": "b", "task": "voice_isolation", "lifecycle": "released"},
        ],
    }
    path = tmp_path / "catalog.yaml"
    path.write_text(yaml.safe_dump(catalog), encoding="utf-8")
    with pytest.raises(ModelZooError, match="duplicate model id"):
        ModelZoo.from_file(path)


def test_hash_mismatch_is_reported(tmp_path):
    zoo = ModelZoo.default()
    model = zoo.get("speaker-verification-ps-spk-v1-1")
    artifact = model.artifacts[0]
    original = artifact.sha256
    # Pydantic models are frozen; use a tiny catalog copy through YAML to test
    # the validator's explicit failure mode without touching repository files.
    raw = yaml.safe_load(zoo.catalog_path.read_text(encoding="utf-8"))
    for item in raw["models"]:
        for key in ("source_checkpoint", "source_config"):
            if item.get(key):
                item[key] = str(zoo.path_for(item[key]))
        for candidate in item["artifacts"]:
            candidate["path"] = str(zoo.path_for(candidate["path"]))
            if candidate.get("manifest"):
                candidate["manifest"] = str(zoo.path_for(candidate["manifest"]))
        if item["id"] == model.id:
            item["artifacts"][0]["sha256"] = "0" * 64
    path = tmp_path / "catalog.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    broken = ModelZoo.from_file(path)
    with pytest.raises(ModelZooValidationError, match="SHA256 mismatch"):
        broken.validate(check_graph=False)
    assert len(original) == hashlib.sha256(zoo.artifact_path(model.id).read_bytes()).digest_size * 2
