"""Tracing a synthesised row: it never changes the row, and the taps follow the
stage registry from the loaded utterance to the emitted pair."""

import numpy as np
import pytest
import torch
import yaml

from puresound.config import load_recipe
from puresound.task.ns import NoiseSuppressionDataset
from puresound.task.trace import STAGE_INDEX, recording
from puresound.task.voice_isolation import VoiceIsolationDataset

SR = 16000


def _recipe(metafile, noise_dir, *, task="noise_suppression", **overrides) -> dict:
    """Every stage the skeleton has, each at probability 1."""
    recipe = {
        "schema_version": 2,
        "purpose": "train",
        "task": task,
        "dataset": {
            "train_metafile": str(metafile),
            "valid_metafile": str(metafile),
            "test_folder": "/tmp/unused",
            "proc_output_folder": "/tmp/unused",
            "target_sample_rate": SR,
            "gain_normalized_to": -28.0,
            "training_length_seconds": 1.0,
            "filter_min_utterance_length": 0.5,
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
            "is_target": False,
            "add_n_cases": 1,
            "snr_range": [0.0, 5.0],
            "media_voice": {"used": True, "prob": 0.5},
            "echo_playback": {"used": True, "prob": 1.0},
            "overlap_control": {"used": True},
        },
        "augmentation_noise": {
            "used": True,
            "prob": 1.0,
            "noise_folder": str(noise_dir),
            "snr_range": [0.0, 10.0],
            "prob_white_noise": 1.0,
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
                "room_dim_range": [[3.0, 6.0], [3.0, 6.0], [2.4, 3.0]],
                "rt60_range": [0.2, 0.6],
                "source_receiver_distance_range": [0.3, 4.0],
                "nsample": 2048,
            },
        },
        "augmentation_speed": {"used": True, "prob": 1.0, "speed_range": [0.9, 1.1]},
        "augmentation_row_initial_ambient": {"used": True, "prob": 1.0, "lead_seconds_range": [0.2, 0.3]},
        "augmentation_src": {"used": True, "prob": 1.0, "src_range": [8000], "prob_each": [1.0]},
        "augmentation_ir_response": {"used": True, "prob": 1.0},
        "augmentation_hpf": {"used": True, "prob": 1.0, "cutoff": [100.0], "prob_each": [1.0]},
        "augmentation_volume": {
            "used": True,
            "prob": 1.0,
            "perturbed_range": [0.5, 0.8],
            "clipping_prob": 0.0,
            "clipping_range": {"min": [0.0, 0.1], "max": [0.9, 1.0]},
        },
        "augmentation_compressor": {
            "used": True,
            "prob": 1.0,
            "threshold_db_range": [-30.0, -20.0],
            "ratio_range": [2.0, 4.0],
            "attack_ms_range": [5.0, 10.0],
            "release_ms_range": [50.0, 100.0],
        },
        "augmentation_codec": {
            "used": True,
            "prob": 1.0,
            "codecs": ["libopus"],
            "bitrate_range": {"libopus": [16000, 16000]},
        },
        "augmentation_packet_loss": {
            "used": True,
            "prob": 1.0,
            "packet_ms_choices": [20],
            "loss_rate_range": [0.05, 0.05],
        },
        "vad_label": {"used": True, "backend": "energy"},
    }
    for key, value in overrides.items():
        recipe[key] = value
    return recipe


def _dataset(tmp_path, recipe: dict):
    path = tmp_path / f"{recipe['task']}_{len(list(tmp_path.iterdir()))}.yaml"
    path.write_text(yaml.safe_dump(recipe, sort_keys=False))
    loaded = load_recipe(path, expected_task=recipe["task"])
    corpus = loaded.dataset
    cls = VoiceIsolationDataset if recipe["task"] == "voice_isolation" else NoiseSuppressionDataset
    return cls(
        metafile_path=corpus.train_metafile,
        min_utt_length_in_seconds=corpus.filter_min_utterance_length,
        min_utts_in_each_speaker=corpus.filter_min_utterance_per_speaker,
        target_sr=corpus.target_sample_rate,
        training_sample_length_in_seconds=corpus.training_length_seconds,
        audio_gain_normalized_to=corpus.gain_normalized_to,
        dataset_role="train",
        pipeline_role="train",
        **loaded.augmentation_kwargs(),
    )


def _assert_same_row(left: dict, right: dict) -> None:
    assert sorted(left) == sorted(right)
    for key in left:
        if torch.is_tensor(left[key]):
            assert torch.equal(left[key], right[key]), key
        elif isinstance(left[key], dict):
            _assert_same_row(left[key], right[key])
        else:
            assert left[key] == right[key], key


@pytest.mark.slow
@pytest.mark.parametrize("task", ["noise_suppression", "voice_isolation"])
def test_tracing_a_row_does_not_change_it(tmp_path, tone_corpus, task):
    metafile, noise_dir = tone_corpus
    overrides = {"augmentation_target_absent": {"used": True, "prob": 0.3, "force_interferer": True}}
    if task == "voice_isolation":
        speech = _recipe(metafile, noise_dir)["augmentation_speech"]
        speech["mix_mode"] = {
            "used": True,
            "modes": [
                {"name": "physical", "prob": 0.5, "physical": True},
                {"name": "moderate", "prob": 0.5, "sir_range": [0.0, 8.0]},
            ],
        }
        overrides["augmentation_speech"] = speech
    dataset = _dataset(tmp_path, _recipe(metafile, noise_dir, task=task, **overrides))
    speakers = sorted(dataset.total_spks)
    for index in range(4):
        key = (speakers[index % len(speakers)], SR, 500 + index)
        plain = dataset[key]
        with recording(dataset) as trace:
            traced = dataset[key]
        _assert_same_row(plain, traced)
        _assert_same_row(plain, dataset[key])
        assert trace.stage_ids()[-1] == "row.emit"


@pytest.mark.slow
def test_taps_follow_the_stage_order_and_end_on_the_emitted_pair(tmp_path, tone_corpus):
    metafile, noise_dir = tone_corpus
    dataset = _dataset(tmp_path, _recipe(metafile, noise_dir))
    speaker = sorted(dataset.total_spks)[0]
    with recording(dataset) as trace:
        sample = dataset[(speaker, SR, 1234)]
    ids = trace.stage_ids()
    positions = [STAGE_INDEX[stage] for stage in ids]
    assert positions == sorted(positions) and len(set(ids)) == len(ids)
    must = {
        "source.load", "row.plan", "foreground.channel", "interferers.sample",
        "interferers.gate", "interferers.mix", "echo.playback",
        "speed.perturb", "ambient.lead", "noise.recorded", "noise.floor", "chain.src",
        "chain.iir", "chain.hpf", "chain.volume", "chain.compressor", "chain.codec",
        "chain.packet_loss", "row.emit",
    }
    assert must <= set(ids)
    # The row sits well below full scale, so the peak guard had nothing to do.
    assert not {"row.target_absent", "reverb.whole_mix", "level.peak_guard"} & set(ids)

    emitted = trace.taps[-1]
    np.testing.assert_array_equal(emitted.noisy, sample["noisy_speech"].reshape(-1).numpy())
    np.testing.assert_array_equal(emitted.target, sample["clean_speech"].reshape(-1).numpy())
    by_stage = {tap.stage: tap for tap in trace.taps}
    assert by_stage["source.load"].params["sample_rate"] == SR
    assert by_stage["row.plan"].params["plan"]["class"] == "RowPlan"
    assert by_stage["interferers.mix"].signals["interferers"].size > 0
    assert by_stage["chain.src"].params["target_sr"] == 8000.0

    roles = {(record.role, ids[record.position]) for record in trace.rirs}
    assert ("foreground", "foreground.channel") in roles
    assert any(stage == "interferers.sample" for _, stage in roles)
    assert any(stage == "echo.playback" for _, stage in roles)
    foreground = next(record for record in trace.rirs if record.role == "foreground")
    assert foreground.impulse is not None and foreground.impulse.size > 0
    assert "source_receiver_distance" in foreground.metadata


@pytest.mark.slow
def test_a_target_absent_row_taps_the_subtraction(tmp_path, tone_corpus):
    metafile, noise_dir = tone_corpus
    recipe = _recipe(
        metafile,
        noise_dir,
        augmentation_target_absent={"used": True, "prob": 1.0, "force_interferer": True},
    )
    dataset = _dataset(tmp_path, recipe)
    with recording(dataset) as trace:
        dataset[(sorted(dataset.total_spks)[0], SR, 99)]
    by_stage = {tap.stage: tap for tap in trace.taps}
    assert by_stage["row.plan"].params["plan"]["target_absent"] is True
    assert not by_stage["row.target_absent"].target.any()
    assert not by_stage["row.emit"].target.any()


@pytest.mark.slow
def test_a_stage_that_runs_without_acting_records_nothing(tmp_path, tone_corpus):
    """Overlap gating with no overlap_control block returns the interferers
    untouched, so the gate stage must not claim it acted."""
    metafile, noise_dir = tone_corpus
    recipe = _recipe(metafile, noise_dir)
    del recipe["augmentation_speech"]["overlap_control"]
    dataset = _dataset(tmp_path, recipe)
    with recording(dataset) as trace:
        dataset[(sorted(dataset.total_spks)[0], SR, 1234)]
    ids = trace.stage_ids()
    assert "interferers.mix" in ids and "interferers.gate" not in ids
