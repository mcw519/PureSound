"""Synthesis invariants that only a fingerprint can see.

Half of `ns.__getitem__`'s correctness is the ORDER it draws randomness in, and
which default fires when a knob is absent. Neither is visible by reading the
code, and neither has a natural assertion -- "the mixture is right" has no
closed form. What does have one:

* **A default belongs to exactly one place.** A recipe that spells every knob
  out and a recipe that spells out only the required ones must synthesise the
  *same* audio, because the config model is the one place a default lives.
  When the two diverge, a default drifted.
* **A seeded item is a pure function of its seed.** The seeded sampler exists so
  validation metrics compare across epochs and runs; if an item is not
  reproducible, that comparison is noise.

For refactors that must not change synthesis at all -- splitting the device
chain, deleting a stage -- these invariants are not enough on their own, because
both configs can move together. Use `tools/rng_fingerprint.py` for that: capture
before, refactor, compare after.
"""

import hashlib

import pytest
import torch
import yaml

from puresound.config import load_recipe
from puresound.task.voice_isolation import VoiceIsolationDataset

SR = 16000
N_ITEMS = 8


def _recipe_dict(metafile, noise_dir, *, spell_out_defaults: bool) -> dict:
    """A voice-isolation recipe with every RNG-gated block on.

    ``spell_out_defaults`` writes every defaulted knob out explicitly. With it
    False those keys are absent, so the model defaults have to stand in for
    them.
    """
    speech = {
        "used": True,
        "prob": 1.0,
        "is_target": False,
        "add_n_cases": [1, 2],
        "snr_range": [-5.0, 10.0],
        "media_voice": {"used": True, "prob": 0.5},
        "echo_playback": {"used": True, "prob": 0.3},
        "overlap_control": {"used": True},
        "mix_mode": {
            "used": True,
            "modes": [
                {"name": "physical", "prob": 0.4, "physical": True},
                {"name": "distance_level", "prob": 0.3, "distance_level": True},
                {"name": "moderate", "prob": 0.3, "sir_range": [0.0, 8.0]},
            ],
        },
    }
    noise = {
        "used": True,
        "prob": 0.9,
        "noise_folder": str(noise_dir),
        "snr_range": [0.0, 20.0],
        "prob_white_noise": 0.5,
        "white_noise_snr_range": [20.0, 40.0],
        "absolute_floor": {"used": True, "prob": 0.5},
    }
    vad_label = {"used": True, "backend": "energy"}

    if spell_out_defaults:
        speech["media_voice"].update(
            hp_cutoff_range=[200.0, 400.0],
            lp_cutoff_range=[3500.0, 7000.0],
            compress_power_range=[0.6, 0.9],
        )
        speech["echo_playback"].update(
            distance_range=[0.2, 1.0], erle_db_range=[20.0, 35.0]
        )
        speech["overlap_control"].update(
            no_overlap_prob=0.25,
            high_overlap_prob=0.25,
            mid_overlap_range=[0.1, 0.5],
            high_overlap_range=[0.5, 1.0],
            fill_on_silent_range=[0.3, 0.5],
            fade_samples=400,
            turn_taking_prob=0.0,
            turn_near_seconds=[1.5, 3.0],
            turn_far_seconds=[2.0, 4.5],
            turn_gap_seconds=[0.0, 0.4],
            turn_overlap_seconds=[0.0, 0.3],
            far_first_prob=0.5,
        )
        speech["mix_mode"]["modes"][1]["jitter_db"] = [-3.0, 3.0]
        noise["absolute_floor"]["level_dbfs_range"] = [-55.0, -35.0]
        vad_label.update(frame_length=400, hop_length=160)

    return {
        "schema_version": 2,
        "purpose": "train",
        "task": "voice_isolation",
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
        "augmentation_speech": speech,
        "augmentation_noise": noise,
        # A real room simulator, not just a folder RIR: source-level reverb is
        # what produces the per-source distance/DRR metadata, and without it the
        # `distance_level` mix mode falls back before it ever reads `jitter_db`.
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
        "augmentation_speed": {"used": True, "prob": 0.5, "speed_range": [0.9, 1.1]},
        "augmentation_src": {
            "used": True,
            "prob": 0.6,
            "src_range": [8000, 24000],
            "prob_each": [0.5, 0.5],
        },
        "augmentation_ir_response": {"used": True, "prob": 0.6},
        "augmentation_hpf": {
            "used": True,
            "prob": 0.6,
            "cutoff": [80.0, 120.0],
            "prob_each": [0.5, 0.5],
        },
        "augmentation_volume": {
            "used": True,
            "prob": 0.6,
            "perturbed_range": [0.5, 1.5],
            "clipping_prob": 0.4,
            "clipping_range": {"min": [0.01, 0.05], "max": [0.95, 0.99]},
        },
        "augmentation_packet_loss": {
            "used": True,
            "prob": 0.5,
            "packet_ms_choices": [20, 60],
            "loss_rate_range": [0.01, 0.1],
        },
        "augmentation_target_absent": {
            "used": True,
            "prob": 0.2,
            "force_interferer": True,
        },
        "vad_label": vad_label,
    }


def _fingerprint(tmp_path, tone_corpus, *, spell_out_defaults: bool) -> str:
    metafile, noise_dir = tone_corpus
    path = tmp_path / f"recipe_{spell_out_defaults}.yaml"
    path.write_text(
        yaml.safe_dump(
            _recipe_dict(metafile, noise_dir, spell_out_defaults=spell_out_defaults),
            sort_keys=False,
        )
    )
    recipe = load_recipe(path, expected_task="voice_isolation")
    corpus_cfg = recipe.dataset
    dataset = VoiceIsolationDataset(
        metafile_path=corpus_cfg.train_metafile,
        min_utt_length_in_seconds=corpus_cfg.filter_min_utterance_length,
        min_utts_in_each_speaker=corpus_cfg.filter_min_utterance_per_speaker,
        target_sr=corpus_cfg.target_sample_rate,
        training_sample_length_in_seconds=corpus_cfg.training_length_seconds,
        audio_gain_normalized_to=corpus_cfg.gain_normalized_to,
        dataset_role="train",
        pipeline_role="train",
        **recipe.augmentation_kwargs(),
    )

    digest = hashlib.sha256()
    speakers = sorted(dataset.total_spks)
    for index in range(N_ITEMS):
        # The 3-tuple carries a per-item seed, the same way the seeded sampler
        # drives deterministic validation.
        sample = dataset[(speakers[index % len(speakers)], SR, 1000 + index)]
        for key in sorted(sample):
            value = sample[key]
            digest.update(key.encode())
            digest.update(
                value.detach().to(torch.float64).contiguous().numpy().tobytes()
                if torch.is_tensor(value)
                else repr(value).encode()
            )
    return digest.hexdigest()


@pytest.mark.slow  # synthesises the fingerprint items three times
def test_a_seeded_item_is_a_pure_function_of_its_seed_and_the_spelled_out_knobs(
    tmp_path, tone_corpus
):
    """Two datasets built from one recipe agree item for item, and spelling
    every knob out synthesises the same audio as omitting them.

    The second is the standing guard on the config-model defaults: set one of
    them wrong -- `fade_samples` to None, `no_overlap_prob` to 0.0 -- and the
    two recipes diverge here while every other test stays green.
    """
    spelled_out = _fingerprint(tmp_path, tone_corpus, spell_out_defaults=True)
    assert _fingerprint(tmp_path, tone_corpus, spell_out_defaults=True) == spelled_out, (
        "a seeded item is not reproducible"
    )
    assert _fingerprint(tmp_path, tone_corpus, spell_out_defaults=False) == spelled_out, (
        "a config-model default differs from its spelled-out value"
    )
