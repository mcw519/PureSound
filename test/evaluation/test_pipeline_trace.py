"""Building one row from chosen audio: the recipe rewrite, the traced report,
and what the inspector refuses before it synthesises anything."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import yaml

from puresound.evaluation import pipeline_trace as pt
from puresound.task.trace import STAGES

SR = 16000


def _tone(seconds, freq, amplitude=0.1):
    time = np.arange(int(SR * seconds)) / SR
    wav = amplitude * np.sin(2 * np.pi * freq * time) + 0.1 * amplitude * np.sin(2 * np.pi * 3.7 * freq * time)
    return wav.astype(np.float32)


def _write(path: Path, samples, rate=SR) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, samples, rate, subtype="FLOAT")
    return path


def _room(folder: Path) -> Path:
    """A five-source bank room: two near and three far channels with geometry."""
    rng = np.random.default_rng(3)
    labels = ["near_0", "near_1", "far_0", "far_1", "far_2"]
    positions = [[2.4, 2.0, 1.2], [1.6, 2.2, 1.1], [4.0, 3.0, 1.5], [0.8, 0.6, 1.4], [4.5, 0.8, 1.2]]
    mic = np.array([2.0, 2.0, 1.0])
    time = np.arange(int(0.4 * SR)) / SR
    channels, channel_map = [], []
    for index, (label, position) in enumerate(zip(labels, positions)):
        distance = float(np.linalg.norm(np.array(position) - mic))
        delay = int(distance / 343.0 * SR)
        rir = np.zeros(time.size)
        rir[delay] = 1.0
        rir[delay + 1 :] += 0.2 * rng.standard_normal(time.size - delay - 1) * np.exp(-6.9 * time[: time.size - delay - 1] / 0.4)
        channels.append(rir / np.max(np.abs(rir)) * 0.98)
        channel_map.append({"channel": index, "label": label, "source_pos": position, "distance_m": distance})
    wav = _write(folder / "room_a.wav", np.stack(channels, axis=1).astype(np.float32))
    wav.with_suffix(".json").write_text(json.dumps({"scene": {
        "room_dim": [5.0, 4.0, 3.0], "rt60": 0.4, "mic_pos": mic.tolist(),
        "source_pos": positions, "source_labels": labels, "channel_map": channel_map,
        "obstacles": [{"footprint": [[1, 1], [1.6, 1], [1.6, 1.5], [1, 1.5]], "z_min": 0.0, "z_max": 0.9, "material": "wood"}],
    }}))
    return wav


@pytest.fixture(scope="module")
def clips(tmp_path_factory):
    root = tmp_path_factory.mktemp("clips")
    rng = np.random.default_rng(1)
    time = np.arange(int(0.3 * SR)) / SR
    decay = np.exp(-6.9 * time / 0.3)
    rir_2ch = np.stack([np.r_[1.0, 0.3 * rng.standard_normal(time.size - 1) * decay[1:]], np.r_[0.5, np.zeros(time.size - 1)]], axis=1)
    return SimpleNamespace(
        foreground=_write(root / "fg.wav", _tone(1.8, 220.0)),
        loud_foreground=_write(root / "fg_loud.wav", _tone(1.8, 220.0, amplitude=0.9)),
        talker_a=[_write(root / "a0.wav", _tone(1.6, 330.0)), _write(root / "a1.wav", _tone(1.6, 350.0))],
        noises=[_write(root / f"n{i}.wav", 0.05 * rng.standard_normal(SR * 2).astype(np.float32)) for i in range(2)],
        long_noise=_write(root / "long.wav", 0.01 * rng.standard_normal(int(SR * 61)).astype(np.float32)),
        silent=_write(root / "silent.wav", np.zeros(SR, dtype=np.float32)),
        not_audio=(root / "notes.wav"),
        rir_2ch=_write(root / "rir.wav", rir_2ch.astype(np.float32)),
        room=_room(root / "rooms"),
    )


def _template(task="noise_suppression", **overrides) -> dict:
    recipe = {
        "schema_version": 2,
        "purpose": "train",
        "task": task,
        "dataset": {
            "train_metafile": "/corpus/train.csv",
            "valid_metafile": "/corpus/valid.csv",
            "test_folder": "/corpus/test",
            "proc_output_folder": "/corpus/proc",
            "target_sample_rate": SR,
            "gain_normalized_to": -28.0,
            "training_length_seconds": 1.5,
            "filter_min_utterance_length": 4.0,
            "filter_min_utterance_per_speaker": 2,
        },
        "trainer": {
            "lightning_trainer_args": {},
            "train_iter_per_epoch": 1,
            "valid_iter_per_epoch": 1,
            "n_spk_per_batch": 2,
            "n_utt_per_speaker": 1,
            "num_workers": 0,
            "num_gpus": 0,
            "work_folder": "/tmp/unused",
        },
        "optimizer": {"type": "Adam", "learning_rate": 0.001},
        "scheduler": {"type": "StepLR", "warmup_step": 0, "args": {"step_size": 1}},
        "loss_func": [{"type": "SDRLoss", "weighted": 1.0}],
        "model": {},
        "augmentation_speech": {
            "used": True,
            "prob": 1.0,
            "is_target": task == "noise_suppression",
            "add_n_cases": 1,
            "snr_range": [5.0, 10.0],
            "echo_playback": {"used": True, "prob": 1.0},
            "overlap_control": {"used": True},
        },
        "augmentation_noise": {
            "used": True,
            "prob": 1.0,
            "noise_folder": "/corpus/noise",
            "snr_range": [0.0, 10.0],
            "prob_white_noise": 0.5,
            "white_noise_snr_range": [20.0, 30.0],
            "absolute_floor": {"used": True, "prob": 1.0},
        },
        "augmentation_reverb": {
            "used": True,
            "prob": 1.0,
            "target_rir_type": "early",
            "simulator": {
                "used": True,
                "source_level": True,
                "pregenerated": {
                    "used": True,
                    "banks": [{
                        "name": "wide", "weight": 1.0, "bank_type": "room", "folder": "/banks/wide",
                        "near_labels": ["near_0", "near_1"], "far_labels": ["far_0", "far_1", "far_2"],
                    }],
                },
            },
        },
        "augmentation_speed": {"used": True, "prob": 1.0, "speed_range": [0.95, 1.05]},
        "augmentation_src": {"used": True, "prob": 1.0, "src_range": [8000], "prob_each": [1.0]},
        "augmentation_hpf": {"used": True, "prob": 1.0, "cutoff": [100.0], "prob_each": [1.0]},
        "augmentation_volume": {
            "used": True, "prob": 1.0, "perturbed_range": [0.5, 0.8], "clipping_prob": 0.0,
            "clipping_range": {"min": [0.0, 0.1], "max": [0.9, 1.0]},
        },
        "augmentation_codec": {"used": True, "prob": 1.0, "codecs": ["libopus"], "bitrate_range": {"libopus": [16000, 16000]}},
        "augmentation_packet_loss": {"used": True, "prob": 1.0, "packet_ms_choices": [20], "loss_rate_range": [0.05, 0.05]},
        "vad_label": {"used": True, "backend": "energy"},
    }
    recipe.update(overrides)
    return recipe


def _recipe_file(tmp_path, template) -> Path:
    path = tmp_path / f"recipe_{len(list(tmp_path.glob('recipe_*')))}.yaml"
    path.write_text(yaml.safe_dump(template, sort_keys=False))
    return path


def _request(clips, recipe, **overrides) -> pt.TraceRequest:
    fields = dict(
        recipe=recipe,
        foreground=clips.foreground,
        talkers=(tuple(clips.talker_a),),
        noises=tuple(clips.noises),
        rir=pt.RirChoice("samples", (clips.room,)),
        seed=7,
    )
    fields.update(overrides)
    return pt.TraceRequest(**fields)


def _trace(tmp_path, request, **kwargs):
    root = tmp_path / f"ws_{len(list(tmp_path.glob('ws_*')))}"
    root.mkdir()
    return pt.trace_row(request, workspace_root=root, **kwargs)


def _audio_refs(report):
    for stage in report["stages"]:
        yield from (stage["audio"] or {}).values()
        if stage["model"] and stage["model"].get("audio"):
            yield stage["model"]["audio"]


def test_explorer_recipe_points_every_corpus_path_at_the_workspace(tmp_path):
    template = _template(
        "voice_isolation",
        augmentation_realfar={"used": True, "pool_manifest": "/pools/far.jsonl", "prob": 0.5},
        augmentation_session_rows={"enabled": True, "prob": 0.5},
        vad_label={"used": True, "backend": "silero", "args": {"threshold": 0.5}},
        curriculum={"used": True, "tracks": [
            {"path": "aug:augmentation_session_rows.prob", "interp": "linear", "points": [[0, 0.0], [8, 0.5]]},
            {"path": "bank:wide", "points": [[0, 1.0], [8, 0.5]]},
            {"path": "loss:SDRLoss", "interp": "linear", "points": [[0, 1.0], [8, 0.5]]},
        ]},
    )
    template["augmentation_noise"]["noise_sources"] = [{"name": "dns", "folder": "/corpus/noise", "weight": 1.0}]
    workspace = pt.Workspace(
        root=tmp_path, metafile=tmp_path / "corpus.csv", noise_dir=tmp_path / "noise",
        rooms_dir=tmp_path / "rooms", rir_dir=tmp_path / "rir",
        noise_files=[tmp_path / "noise" / "n.wav"], room_files=[tmp_path / "rooms" / "items" / "r.wav"],
        rir_files=[tmp_path / "rir" / "r.wav"], talker_count=1,
    )
    recipe, notes = pt.explorer_recipe(template, workspace, pt.RirChoice("samples", (Path("r.wav"),)))
    corpus = recipe["dataset"]
    assert corpus["train_metafile"] == corpus["valid_metafile"] == str(tmp_path / "corpus.csv")
    assert (corpus["filter_min_utterance_per_speaker"], corpus["target_sample_rate"]) == (1, SR)
    assert recipe["augmentation_noise"]["noise_folder"] == str(tmp_path / "noise")
    assert "noise_sources" not in recipe["augmentation_noise"]
    banks = recipe["augmentation_reverb"]["simulator"]["pregenerated"]["banks"]
    assert [(bank["name"], bank["folder"], bank["far_labels"]) for bank in banks] == [("samples", str(tmp_path / "rooms"), ["far_0", "far_1", "far_2"])]
    assert recipe["augmentation_realfar"]["used"] is False
    assert recipe["augmentation_session_rows"]["enabled"] is False
    assert [track["path"] for track in recipe["curriculum"]["tracks"]] == ["loss:SDRLoss"]
    assert recipe["vad_label"] == {"used": True, "backend": "energy"}
    subjects = {note["subject"] for note in notes}
    assert {"augmentation_realfar", "augmentation_session_rows", "curriculum bank:wide",
            "curriculum aug:augmentation_session_rows.prob", "vad_label"} <= subjects

    uploaded, notes = pt.explorer_recipe(template, workspace, pt.RirChoice("upload", (Path("r.wav"),)))
    assert uploaded["augmentation_reverb"]["rir_folder"] == str(tmp_path / "rir")
    assert uploaded["augmentation_reverb"]["simulator"] is None
    assert any("whole mixture" in note["reason"] for note in notes)
    dry, _ = pt.explorer_recipe(template, workspace, pt.RirChoice("none"))
    assert dry["augmentation_reverb"]["used"] is False
    assert template["dataset"]["train_metafile"] == "/corpus/train.csv"  # the template is not mutated


@pytest.mark.slow
def test_a_traced_row_reports_every_stage_in_order_as_strict_json(tmp_path, clips):
    result = _trace(tmp_path, _request(clips, _recipe_file(tmp_path, _template()), epoch=5))
    report, audio = result["report"], result["audio"]
    json.dumps(report, allow_nan=False)
    assert report["schema"] == pt.SCHEMA and report["task"] == "noise_suppression"
    assert [stage["id"] for stage in report["stages"]] == [spec.id for spec in STAGES]
    assert set(_audio_refs(report)) <= set(audio)
    stages = {stage["id"]: stage for stage in report["stages"]}
    for stage_id in ("source.load", "foreground.channel", "interferers.mix", "noise.recorded", "chain.codec", "row.emit"):
        assert stages[stage_id]["fired"], stage_id
    assert not stages["reverb.whole_mix"]["fired"] and stages["reverb.whole_mix"]["audio"] is None
    assert stages["chain.codec"]["recipe"] == {"configured": True, "enabled": True, "prob": 1.0}
    assert stages["chain.codec"]["block"] == "augmentation_codec_args" and stages["row.emit"]["block"] is None
    assert stages["foreground.channel"]["metrics"]["esnr_state"] == "finite"  # late reverb counts as residual
    assert stages["source.load"]["metrics"]["esnr_state"] == "identical"
    assert stages["foreground.channel"]["rirs"]
    rir = report["rirs"][stages["foreground.channel"]["rirs"][0]]
    assert rir["role"] == "foreground" and set(rir["modes"]) == {"full", "early"} and rir["summary"]["edc_db"]
    assert report["room"]["kind"] == "box"
    assert any("foreground" in source["roles"] for source in report["room"]["sources"])
    assert report["epoch"] is None and any(note["subject"] == "epoch" for note in report["notes"])
    assert report["emitted"]["codec_applied"] == 1.0
    assert all(stage["model"] is None for stage in report["stages"])
    assert report["row_seconds"] == pytest.approx(1.5, abs=0.01)


@pytest.mark.slow
def test_the_same_request_rebuilds_the_same_row(tmp_path, clips):
    recipe = _recipe_file(tmp_path, _template())

    def emitted(seed):
        result = _trace(tmp_path, _request(clips, recipe, seed=seed))
        stage = next(stage for stage in result["report"]["stages"] if stage["id"] == "row.emit")
        return result["audio"][stage["audio"]["noisy"]]

    first = emitted(11)
    np.testing.assert_array_equal(first, emitted(11))
    assert not np.array_equal(first, emitted(12))


@pytest.mark.slow
def test_every_stage_is_scored_on_the_gain_staged_pair(tmp_path, clips):
    template = _template(
        dataset={**_template()["dataset"], "gain_normalized_to": None},
        augmentation_volume={"used": True, "prob": 1.0, "perturbed_range": [3.0, 3.0], "clipping_prob": 0.0,
                             "clipping_range": {"min": [0.0, 0.1], "max": [0.9, 1.0]}},
    )
    seen = []

    def enhance(noisy, rate):
        assert rate == SR
        seen.append(float(np.max(np.abs(noisy))))
        return 0.5 * noisy

    result = _trace(
        tmp_path,
        _request(clips, _recipe_file(tmp_path, template), foreground=clips.loud_foreground),
        enhance=enhance,
        model={"id": "fake", "sample_rate": SR},
    )
    report, audio = result["report"], result["audio"]
    stages = {stage["id"]: stage for stage in report["stages"]}
    assert max(seen) <= 1.0 + 1e-6
    assert stages["level.peak_guard"]["fired"] and stages["level.peak_guard"]["changed"]
    assert float(np.max(np.abs(audio[stages["chain.volume"]["audio"]["noisy"]]))) > 1.0
    volume = stages["chain.volume"]["model"]
    assert volume["gain"] < 1.0
    assert volume["output"]["si_sdr_db"] == pytest.approx(volume["input"]["si_sdr_db"], abs=1e-3)
    assert volume["level_change_db"] == pytest.approx(-6.02, abs=0.05)
    assert any("same_input_as" in (stage["model"] or {}) for stage in report["stages"])
    assert report["model"] == {"id": "fake", "sample_rate": SR}
    json.dumps(report, allow_nan=False)


@pytest.mark.slow
def test_a_target_absent_row_reports_no_target_instead_of_a_ratio(tmp_path, clips):
    template = _template(augmentation_target_absent={"used": True, "prob": 1.0, "force_interferer": True})
    template["augmentation_speech"]["is_target"] = False
    result = _trace(tmp_path, _request(clips, _recipe_file(tmp_path, template)), enhance=lambda noisy, rate: 0.1 * noisy)
    stages = {stage["id"]: stage for stage in result["report"]["stages"]}
    emitted = stages["row.emit"]
    assert emitted["metrics"]["esnr_state"] == "no_target" and emitted["metrics"]["esnr_db"] is None
    assert emitted["model"]["output"]["si_sdr_db"] is None
    assert emitted["model"]["level_change_db"] == pytest.approx(-20.0, abs=0.1)
    assert stages["row.target_absent"]["fired"]
    json.dumps(result["report"], allow_nan=False)


@pytest.mark.slow
def test_blocks_without_a_corpus_are_switched_off_and_the_rest_still_runs(tmp_path, clips):
    template = _template(
        "voice_isolation",
        augmentation_realfar={"used": True, "pool_manifest": "/pools/far.jsonl", "prob": 0.5},
        augmentation_session_rows={"enabled": True, "prob": 0.5},
        curriculum={"used": True, "tracks": [
            {"path": "aug:augmentation_session_rows.prob", "interp": "linear", "points": [[0, 0.0], [8, 0.5]]},
            {"path": "bank:wide", "points": [[0, 1.0], [8, 0.5]]},
            {"path": "loss:SDRLoss", "interp": "linear", "points": [[0, 1.0], [8, 0.5]]},
        ]},
    )
    report = _trace(tmp_path, _request(clips, _recipe_file(tmp_path, template), epoch=3))["report"]
    assert report["task"] == "voice_isolation" and report["curriculum"] is True and report["epoch"] == 3
    subjects = {note["subject"] for note in report["notes"]}
    assert {"augmentation_realfar", "augmentation_session_rows", "curriculum bank:wide"} <= subjects
    assert next(stage for stage in report["stages"] if stage["id"] == "row.emit")["fired"]


@pytest.mark.slow
def test_an_uploaded_impulse_response_reverberates_the_whole_mixture(tmp_path, clips):
    template = _template()
    template["augmentation_reverb"]["simulator"]["source_level"] = False
    result = _trace(tmp_path, _request(clips, _recipe_file(tmp_path, template), rir=pt.RirChoice("upload", (clips.rir_2ch,))))
    stages = {stage["id"]: stage for stage in result["report"]["stages"]}
    assert stages["reverb.whole_mix"]["fired"] and not stages["foreground.channel"]["fired"]
    assert result["report"]["room"] is None
    assert any("whole mixture" in note["reason"] for note in result["report"]["notes"])


def test_inputs_are_refused_before_anything_is_synthesised(tmp_path, clips):
    recipe = _recipe_file(tmp_path, _template())
    clips.not_audio.write_text("not audio")
    cases = [
        (dict(foreground=clips.silent), "silent"),
        (dict(talkers=()), "other talker"),
        (dict(seconds=0.2), "row length"),
        (dict(foreground=clips.not_audio), "could not read"),
    ]
    for overrides, message in cases:
        with pytest.raises(pt.PipelineInputError, match=message):
            _trace(tmp_path, _request(clips, recipe, **overrides))
    workspace = pt.prepare_workspace(_request(clips, recipe, noises=(clips.long_noise,)), tmp_path / "long")
    assert sf.info(workspace.noise_files[0]).frames == int(pt.MAX_INPUT_SECONDS * SR)
    assert any("60 s" in note["reason"] for note in workspace.notes)


def test_a_single_noise_clip_is_enough_for_rows_that_splice_two(tmp_path, clips):
    """A dynamic-noise row splices two different clips from the pool; one chosen
    clip must not make that draw fail."""
    recipe = _recipe_file(tmp_path, _template())
    workspace = pt.prepare_workspace(_request(clips, recipe, noises=(clips.noises[0],)), tmp_path / "one")
    assert len(workspace.noise_files) == 2
    np.testing.assert_array_equal(sf.read(workspace.noise_files[0])[0], sf.read(workspace.noise_files[1])[0])
    assert any("splice" in note["reason"] for note in workspace.notes)


def test_a_response_used_in_two_stages_is_listed_under_each():
    """A far channel can serve an interferer and colour the noise through the
    room; the noise stage has to find it too."""
    from types import SimpleNamespace

    from puresound.task.trace import RirRecord, SynthesisTrace

    trace = SynthesisTrace()
    impulse = np.r_[1.0, 0.5, 0.25].astype(np.float32)
    record = lambda position: RirRecord(position, "interferer", "full", "bank-1", SR, {"label": "far_0"}, impulse)
    trace.tap("source.load", np.ones(4), np.ones(4))
    trace.rirs.append(record(1))
    trace.tap("interferers.sample", np.ones(4), np.ones(4))
    trace.rirs.append(record(2))
    trace.tap("noise.recorded", np.ones(4), np.ones(4))
    stages, rirs = pt._assemble(trace, SimpleNamespace(), pt._AudioStore())
    assert [(rir["stage"], rir["rir_id"]) for rir in rirs] == [("interferers.sample", "bank-1"), ("noise.recorded", "bank-1")]
    assert {stage["id"]: stage["rirs"] for stage in stages}["noise.recorded"] == [1]


def test_notes_say_what_actually_happened(tmp_path, clips):
    workspace = pt.Workspace(
        root=tmp_path, metafile=tmp_path / "corpus.csv", noise_dir=tmp_path / "noise",
        rooms_dir=tmp_path / "rooms", rir_dir=tmp_path / "rir", noise_files=[tmp_path / "n.wav"],
        room_files=[tmp_path / "r.wav"], talker_count=1,
    )
    track = {"used": True, "tracks": [{"path": "bank:wide", "points": [[0, 1.0], [8, 0.5]]}]}
    reasons = lambda notes: {note["subject"]: note["reason"] for note in notes}
    _, notes = pt.explorer_recipe(_template(curriculum=track), workspace, pt.RirChoice("samples", (Path("r.wav"),)))
    assert "sample rooms replace" in reasons(notes)["curriculum bank:wide"]
    _, notes = pt.explorer_recipe(_template(curriculum=track), workspace, pt.RirChoice("none"))
    assert "does not use" in reasons(notes)["curriculum bank:wide"]
    quiet = _template(augmentation_noise={"used": False}, augmentation_reverb={"used": False})
    _, notes = pt.explorer_recipe(quiet, workspace, pt.RirChoice("samples", (Path("r.wav"),)))
    assert "not used" in reasons(notes)["augmentation_noise"] and "not used" in reasons(notes)["augmentation_reverb"]
    silent_rir = pt.Workspace(**{**workspace.__dict__, "notes": [], "rir_files": []})
    _, notes = pt.explorer_recipe(_template(), silent_rir, pt.RirChoice("upload", (Path("silent.wav"),)))
    assert "silent" in reasons(notes)["augmentation_reverb"]
    recipe = _recipe_file(tmp_path, _template())
    with pytest.raises(pt.PipelineInputError, match="silent or shorter"):
        _trace(tmp_path, _request(clips, recipe, talkers=((clips.silent,),)))


def test_a_second_trace_says_it_is_waiting_for_the_first(tmp_path):
    root = tmp_path
    from threading import Thread

    phases = []
    failures = []
    request = pt.TraceRequest(recipe=Path("unused.yaml"), foreground=Path("missing.wav"))

    def run():
        try:
            pt.trace_row(request, workspace_root=root, progress=lambda value, phase: phases.append(phase))
        except pt.PipelineInputError as exc:
            failures.append(str(exc))

    with pt._LOCK:
        worker = Thread(target=run)
        worker.start()
        worker.join(timeout=0.5)
        assert worker.is_alive() and phases == ["waiting for the row being built"]
    worker.join(timeout=10)
    assert failures and "could not read" in failures[0]
