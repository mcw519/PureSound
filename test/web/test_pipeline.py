"""The pipeline inspector's catalog: which recipes it offers and what a request
may name. The trace jobs themselves are in test_server.py."""

import pytest

from puresound.web import WebService
from puresound.web.pipeline import PipelineCatalog, PipelineRequestError, list_recipes


def test_only_released_training_recipes_of_the_traced_tasks_are_offered(pipeline_repo):
    repo = pipeline_repo
    recipes = list_recipes(repo)
    assert [item["id"] for item in recipes] == ["egs/ns/config/train_a.yaml", "egs/vi/config/train_b.yaml"]
    assert recipes[0]["row_seconds"] == 6.0 and recipes[1]["curriculum"] is True
    assert list_recipes(None) == []


def _catalog(repo):
    return PipelineCatalog(repo, WebService().static_dir / "samples" / "pipeline")


def test_a_request_is_built_from_samples_and_uploads(pipeline_repo, tmp_path):
    repo = pipeline_repo
    upload = tmp_path / "upload.wav"
    upload.write_bytes(b"x")
    catalog = _catalog(repo)
    room = catalog.samples()["rooms"][0]["id"]
    request, options = catalog.build_request(
        {
            "recipe": "egs/ns/config/train_a.yaml",
            "seed": 42,
            "epoch": 3,
            "seconds": 4.0,
            "foreground": {"sample": "reader-1272"},
            "talkers": [{"sample": "reader-61"}, {"upload_id": "a" * 32}],
            "noises": [{"sample": "babble"}, {"upload_id": "a" * 32}],
            "rir": {"kind": "samples", "rooms": [room]},
            "model_id": "noise-suppression-dpcrn-mamba-v2",
        },
        lambda upload_id: upload if upload_id == "a" * 32 else None,
    )
    assert request.recipe == repo / "egs/ns/config/train_a.yaml"
    assert request.foreground.name == "speaker-a-1.wav"
    assert [len(files) for files in request.talkers] == [1, 1] and request.talkers[1] == (upload,)
    assert request.noises[0].name == "babble.wav" and request.noises[1] == upload
    assert request.rir.kind == "samples" and request.rir.paths[0].stem == room and request.rir.paths[0].is_file()
    assert (request.seed, request.epoch, request.seconds, request.role) == (42, 3, 4.0, "train")
    assert options == {"model_id": "noise-suppression-dpcrn-mamba-v2", "provider": "cpu", "recipe": "egs/ns/config/train_a.yaml"}


@pytest.mark.parametrize(
    "change, message",
    [
        ({"recipe": "egs/ns/config/exp/train_x.yaml"}, "offered training recipes"),
        ({"recipe": "../../etc/passwd"}, "offered training recipes"),
        ({"recipe": ["egs/ns/config/train_a.yaml"]}, "offered training recipes"),
        ({"recipe": {"id": "egs/ns/config/train_a.yaml"}}, "offered training recipes"),
        ({"role": "test"}, "role"),
        ({"seed": -1}, "seed"),
        ({"seed": True}, "seed"),
        ({"seed": "7"}, "seed"),
        ({"epoch": -2}, "epoch"),
        ({"seconds": 0.5}, "seconds"),
        ({"seconds": 45}, "seconds"),
        ({"foreground": None}, "foreground"),
        ({"foreground": {"sample": "nobody"}}, "sample"),
        ({"foreground": {"upload_id": "b" * 32}}, "upload the file again"),
        ({"noises": [{"sample": "babble"}] * 17}, "at most 16"),
        ({"rir": {"kind": "room-simulator"}}, "rir.kind"),
        ({"rir": {"kind": "samples", "rooms": ["../../x"]}}, "sample"),
    ],
)
def test_a_request_naming_something_not_offered_is_refused(pipeline_repo, change, message):
    repo = pipeline_repo
    payload = {"recipe": "egs/ns/config/train_a.yaml", "foreground": {"sample": "reader-61"}, **change}
    with pytest.raises(PipelineRequestError, match=message):
        _catalog(repo).build_request(payload, lambda upload_id: None)




def test_the_recipe_list_rereads_only_files_that_changed(pipeline_repo, monkeypatch):
    import os

    import puresound.web.pipeline as web_pipeline

    reads = []
    real = web_pipeline.yaml.safe_load
    monkeypatch.setattr(web_pipeline.yaml, "safe_load", lambda text: reads.append(1) or real(text))
    first = list_recipes(pipeline_repo)
    count = len(reads)
    assert list_recipes(pipeline_repo) == first and len(reads) == count
    changed = pipeline_repo / "egs/ns/config/train_a.yaml"
    changed.write_text(changed.read_text().replace("6.0", "4.0"))
    os.utime(changed, ns=(changed.stat().st_atime_ns, changed.stat().st_mtime_ns + 1_000_000))
    assert list_recipes(pipeline_repo)[0]["row_seconds"] == 4.0 and len(reads) == count + 1
