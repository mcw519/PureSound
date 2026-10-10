"""The model zoo catalog: the checked-in inventory, its validation, and the
failures a broken catalog must raise."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from puresound.inference import ModelZoo, ModelZooError, ModelZooValidationError


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_checked_in_catalog_has_expected_inventory_and_defaults():
    zoo = ModelZoo.default()
    assert zoo.get("voice-isolate-dpcrn-curriculum-v1").roles == ["default"]
    assert zoo.get("voice-isolate-dpcrn-curriculum-v2").roles == ["candidate"]
    assert zoo.get("voice-isolate-dpcrn-curriculum-v2").lifecycle == "released"
    assert zoo.default_model("voice_isolation").id == "voice-isolate-dpcrn-curriculum-v1"
    assert zoo.get("speaker-verification-ps-spk-v1-1").roles == ["default"]
    assert zoo.get("noise-suppression-dpcrn-mamba-v2").roles == ["default"]
    assert zoo.get("noise-suppression-dpcrn-mamba-v2").lifecycle == "released"
    assert zoo.get("noise-suppression-dpcrn-mamba-v1").roles == ["candidate"]
    assert zoo.get("noise-suppression-dpcrn-mamba-v3").roles == ["candidate"]
    assert zoo.get("noise-suppression-dpcrn-mamba-v3").lifecycle == "released"
    assert zoo.default_model("ns").id == "noise-suppression-dpcrn-mamba-v2"
    assert [m.id for m in zoo.list(task="ns")] == [
        "noise-suppression-dpcrn-mamba-v1", "noise-suppression-dpcrn-mamba-v2",
        "noise-suppression-dpcrn-mamba-v3"]
    assert zoo.list(task="tse") == []
    # A checkpoint stem resolves to its catalog entry like the model id does.
    assert zoo.get("dpcrn_mamba_v1").id == "noise-suppression-dpcrn-mamba-v1"
    assert zoo.get("dpcrn_mamba_v2").id == "noise-suppression-dpcrn-mamba-v2"
    assert zoo.get("dpcrn_mamba_v3").id == "noise-suppression-dpcrn-mamba-v3"
    assert zoo.get("dpcrn_curriculum_v2").id == "voice-isolate-dpcrn-curriculum-v2"


def test_checked_in_catalog_validates_paths_hashes_sidecars_and_graphs():
    report = ModelZoo.default().validate()
    assert report.ok


def test_a_broken_catalog_is_refused(tmp_path):
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

    # A checked-in artifact whose recorded hash does not match fails validation.
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


def test_native_companion_digests_and_paths_are_validated(tmp_path):
    zoo = ModelZoo.default()
    model = zoo.get("noise-suppression-dpcrn-mamba-v3")
    raw = {"schema_version": 1, "models": [model.model_dump()]}
    item = raw["models"][0]
    for key in ("source_checkpoint", "source_config"):
        item[key] = str(zoo.path_for(item[key]))
    artifact = item["artifacts"][0]
    artifact["path"] = str(zoo.path_for(artifact["path"]))
    manifest = json.loads(zoo.path_for(artifact["manifest"]).read_text())
    # Native paths resolve beside the ONNX, independently of sidecar location.
    assert manifest["cpu_optimization"]["native_graph"].endswith(".native.onnx")
    sidecar = tmp_path / "model.json"
    artifact["manifest"] = str(sidecar)
    catalog = tmp_path / "catalog.yaml"
    catalog.write_text(yaml.safe_dump(raw))
    manifest["cpu_optimization"]["native_sha256"] = "0" * 64
    sidecar.write_text(json.dumps(manifest))
    with pytest.raises(ModelZooValidationError, match="native companion SHA256"):
        ModelZoo.from_file(catalog).validate(check_graph=False)
    manifest["cpu_optimization"]["native_graph"] = "../elsewhere.onnx"
    sidecar.write_text(json.dumps(manifest))
    with pytest.raises(ModelZooValidationError, match="sibling filename"):
        ModelZoo.from_file(catalog).validate(check_graph=False)
    manifest["cpu_optimization"]["native_graph"] = "missing.native.onnx"
    sidecar.write_text(json.dumps(manifest))
    with pytest.raises(ModelZooValidationError, match="native companion not found"):
        ModelZoo.from_file(catalog).validate(check_graph=False)
