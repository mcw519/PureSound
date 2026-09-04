"""``augmentation_session_rows`` -- the multi-turn conversational row type.

What is being guarded here, in the order the deliverable states it:

1. **Off costs nothing.** The block absent, disabled, or at ``prob: 0`` must give
   a bit-identical row, and so must a row shorter than ``min_seconds`` with the
   block fully on -- eligibility is tested before the probability draw, which is
   what makes the short length buckets of a session recipe comparable to the
   same recipe without it.
2. **The labels are on the ``vad_target`` grid.** ``user_active`` IS the energy
   VAD of the dry user signal (the same call ``create_vad_target`` makes),
   ``turn_id`` is contiguous per turn, and double-talk frames are 0.
3. **A user gap is not a target-absent row**, and it is not digital silence.
4. **One device-chain draw per row**, applied jointly, so with the chain
   disabled the mixture is exactly the user plus the bystander bus plus the
   floor.
5. **Collate pads** the per-frame and per-turn labels, including a batch that
   mixes session rows with ordinary ones.
6. **A move is a move, not a new talker**: every user turn keeps one speaker id.
"""

import numpy as np
import pytest
import soundfile as sf
import torch

from puresound.config.augmentation import SessionRowsConfig
from puresound.task.session_rows import (
    ROLE_BYSTANDER,
    ROLE_USER,
    build_session_script,
    collate_session_labels,
    script_masks,
    source_rng_scope,
)
from puresound.task.voice_isolation import (
    VoiceIsolationCollateFunc,
    VoiceIsolationDataset,
)


SR = 16000
HOP = 160
FPS = SR / HOP


# --------------------------------------------------------------------------- #
# corpus + dataset helpers
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    """A long-enough tone corpus: sessions need rows of many seconds.

    Amplitude-modulated so the energy VAD sees activity and silence rather than
    one unbroken active stretch -- the labels are read against it.
    """
    root = tmp_path_factory.mktemp("session_corpus")
    rows = ["uttid, spkid, gender, path, length, sample rate, channels"]
    for speaker in range(6):
        for utterance in range(2):
            path = root / f"spk{speaker}_{utterance}.wav"
            time = np.arange(int(SR * 24.0)) / SR
            wav = (
                0.05
                * np.sin(2 * np.pi * (150.0 + 40.0 * speaker) * time)
                * (0.5 + 0.5 * np.sin(2 * np.pi * (1.5 + 0.2 * utterance) * time))
            )
            sf.write(str(path), wav.astype("float32"), SR)
            rows.append(
                f"spk{speaker}_{utterance}, spk{speaker}, m, {path}, "
                f"{int(SR * 24.0)}, {SR}, 1"
            )
    meta = root / "meta.csv"
    meta.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return meta


_SPEECH_ARGS = {
    "used": True,
    "is_target": False,
    "prob": 0.7,
    "add_n_cases": [1, 2],
    "snr_range": [-10.0, 10.0],
    "overlap_control": {"used": True, "turn_taking_prob": 0.3},
}

#: The on-the-fly shoebox simulator with a short RIR: near/far distance ranges
#: are honoured per source role, which is what the move and the matched
#: bystander need, and 2048 taps keeps it fast.
_REVERB_ARGS = {
    "used": True,
    "prob": 1.0,
    "target_rir_type": "early",
    "simulator": {
        "used": True,
        "source_level": True,
        "room_dim_range": [[3.0, 5.0], [3.0, 5.0], [2.4, 3.0]],
        "rt60_range": [0.2, 0.5],
        "source_receiver_distance_range": [0.3, 4.0],
        "foreground_distance_range": [0.3, 1.0],
        "interferer_distance_range": [1.5, 4.0],
        "nsample": 2048,
    },
}

_VAD_ARGS = {
    "used": True,
    "backend": "energy",
    "args": {"frame_length": 400, "hop_length": 160},
}


def _dataset(corpus, session=None, *, reverb=True, noise=None, seconds=15.0):
    return VoiceIsolationDataset(
        metafile_path=str(corpus),
        min_utt_length_in_seconds=1.0,
        min_utts_in_each_speaker=1,
        target_sr=SR,
        training_sample_length_in_seconds=seconds,
        audio_gain_normalized_to=-28,
        augmentation_speech_args=_SPEECH_ARGS,
        augmentation_reverb_args=_REVERB_ARGS if reverb else None,
        augmentation_noise_args=noise,
        augmentation_session_rows_args=session,
        vad_label_args=_VAD_ARGS,
    )


def _session(**overrides):
    args = {"enabled": True, "prob": 1.0, "min_seconds": 6.0}
    args.update(overrides)
    return args


def _row(dataset, seed, seconds=None):
    """One deterministic item: a 3/4-tuple reseeds every stream the path uses."""
    if seconds is None:
        return dataset[("spk0", SR, seed)]
    return dataset[("spk0", SR, seed, seconds)]


def _runs(mask: np.ndarray, value: int):
    """[(start, end_exclusive)] runs where mask == value."""
    out, index, n = [], 0, len(mask)
    while index < n:
        if mask[index] == value:
            end = index
            while end < n and mask[end] == value:
                end += 1
            out.append((index, end))
            index = end
        else:
            index += 1
    return out


# --------------------------------------------------------------------------- #
# 1. the block costs nothing when it is off
# --------------------------------------------------------------------------- #


def test_schema_defaults_are_off():
    config = SessionRowsConfig()
    assert config.enabled is False
    assert config.prob == 0.0
    assert config.used is False


def test_schema_rejects_a_session_it_cannot_render():
    with pytest.raises(Exception):
        SessionRowsConfig(enabled=True, prob=0.5, min_seconds=30.0, max_seconds=10.0)
    with pytest.raises(Exception):
        SessionRowsConfig(enabled=True, prob=0.5, n_bystanders=[0, 2])
    with pytest.raises(Exception):
        SessionRowsConfig(enabled=True, prob=0.5, user_gap_seconds=[0.2, 5.0])
    # ...and accepts the R1a shape.
    SessionRowsConfig(enabled=True, prob=0.5, min_seconds=12.0, max_seconds=60.0)


def test_the_recipe_refuses_the_two_blocks_a_session_cannot_survive():
    from puresound.config.recipe import VoiceIsolationRecipe

    base = {
        "schema_version": 2,
        "purpose": "train",
        "task": "voice_isolation",
        "dataset": {
            "train_metafile": "a",
            "valid_metafile": "b",
            "test_folder": "c",
            "proc_output_folder": "d",
            "target_sample_rate": 16000,
            "gain_normalized_to": -28.0,
            "training_length_seconds": 6.0,
            "filter_min_utterance_length": 2.0,
            "filter_min_utterance_per_speaker": 1,
        },
        "trainer": {
            "lightning_trainer_args": {},
            "train_iter_per_epoch": 1,
            "valid_iter_per_epoch": 1,
            "n_spk_per_batch": 2,
            "n_utt_per_speaker": 1,
            "num_workers": 0,
            "num_gpus": 0,
            "work_folder": "w",
        },
        "optimizer": {"type": "AdamW", "learning_rate": 0.001},
        "scheduler": {"type": "CosineAnnealingWarmRestarts", "warmup_step": 1},
        "model": {},
        "loss_func": [{"type": "SDRLoss", "weighted": 1.0}],
        "augmentation_session_rows": {"enabled": True, "prob": 0.5},
    }
    VoiceIsolationRecipe.model_validate(base)

    with pytest.raises(Exception, match="is_target"):
        VoiceIsolationRecipe.model_validate(
            {**base, "augmentation_speech": {**_SPEECH_ARGS, "is_target": True}}
        )
    with pytest.raises(Exception, match="row_initial_ambient"):
        VoiceIsolationRecipe.model_validate(
            {
                **base,
                "augmentation_row_initial_ambient": {"used": True, "prob": 0.3},
            }
        )


@pytest.mark.parametrize(
    "disabled",
    [
        None,
        {"enabled": False, "prob": 1.0, "min_seconds": 1.0},
        {"enabled": True, "prob": 0.0, "min_seconds": 1.0},
    ],
)
def test_a_disabled_block_is_bit_identical(corpus, disabled):
    """Absent, disabled and prob-0 must all leave the RNG stream where it was."""
    baseline = _dataset(corpus, None)
    candidate = _dataset(corpus, disabled)
    for seed in range(24):
        left, right = _row(baseline, 1000 + seed), _row(candidate, 1000 + seed)
        assert torch.equal(left["noisy_speech"], right["noisy_speech"]), seed
        assert torch.equal(left["clean_speech"], right["clean_speech"]), seed
        assert torch.equal(left["vad_target"], right["vad_target"]), seed


def test_a_row_below_min_seconds_is_bit_identical_with_the_block_on(corpus):
    """Eligibility before the draw: the short buckets of a session recipe are
    the same rows the recipe without it produces."""
    baseline = _dataset(corpus, None)
    session = _dataset(corpus, _session(prob=1.0, min_seconds=30.0))
    for seed in range(24):
        left = _row(baseline, 2000 + seed, seconds=6.0)
        right = _row(session, 2000 + seed, seconds=6.0)
        assert torch.equal(left["noisy_speech"], right["noisy_speech"]), seed
        assert torch.equal(left["clean_speech"], right["clean_speech"]), seed


def test_an_enabled_block_emits_the_label_contract_on_every_row(corpus):
    """Including the rows that are not sessions: a key present on some rows only
    collates into a tensor shorter than the batch."""
    dataset = _dataset(corpus, _session(prob=1.0, min_seconds=30.0))
    sample = _row(dataset, 7, seconds=6.0)  # too short to be a session
    assert float(sample["session_row"]) == 0.0
    assert float(sample["row_source_id"]) == -1
    assert sample["turn_id"].shape == sample["user_active"].shape
    assert sample["turn_id"].abs().sum() == 0
    assert sample["turn_role"].numel() == 0


# --------------------------------------------------------------------------- #
# 2. the labels
# --------------------------------------------------------------------------- #


def test_user_active_is_the_energy_vad_of_the_dry_user_signal(corpus):
    """The same grid and the same labeler `create_vad_target` uses, which for
    the energy backend makes the two tensors equal frame for frame."""
    dataset = _dataset(corpus, _session())
    for seed in range(6):
        sample = _row(dataset, 3000 + seed)
        assert float(sample["session_row"]) == 1.0
        assert torch.equal(sample["user_active"], sample["vad_target"].reshape(-1))
        assert sample["user_active"].shape == sample["turn_id"].shape
        assert 0.0 < float(sample["user_active"].mean()) < 1.0


def test_turn_id_is_contiguous_and_zero_on_double_talk(corpus):
    dataset = _dataset(corpus, _session(boundary_overlap_prob=1.0))
    for seed in range(8):
        sample = _row(dataset, 4000 + seed)
        turn_id = sample["turn_id"].numpy()
        roles = sample["turn_role"].numpy()
        assert turn_id.max() <= len(roles)
        for k in range(1, int(turn_id.max()) + 1):
            runs = _runs(turn_id, k)
            assert len(runs) <= 1, (seed, k, runs)
        # every non-zero id belongs to a turn whose role is a real role
        for k in np.unique(turn_id):
            if k == 0:
                continue
            assert roles[k - 1] in (ROLE_USER, ROLE_BYSTANDER)
        # Double talk reads as no turn. Measured against the ENERGY VAD's
        # both-active frames, which are a superset of the script's double talk
        # (a raised-cosine turn edge and a reverb tail keep the leaving talker
        # audible for a few frames after its span ends, and those frames do
        # legitimately carry the arriving talker's id): 0.939 of them carry no
        # id at the p0 of 20 rows, 0.979 at the median. The exact contract --
        # ZERO scripted double-talk frames carry an id -- is asserted on the
        # generator itself in `test_turn_id_is_zero_on_exactly_the_scripted_
        # double_talk`, where the script is visible.
        both = (sample["user_active"] > 0) & (sample["bystander_active"] > 0)
        if bool(both.any()):
            overlap_ids = turn_id[both.numpy()]
            share = float((overlap_ids == 0).mean())
            assert share >= 0.90, (seed, share)


def test_turn_vectors_agree_with_the_turn_ids(corpus):
    dataset = _dataset(corpus, _session())
    sample = _row(dataset, 4100)
    roles = sample["turn_role"]
    speakers = sample["turn_speaker"]
    chains = sample["turn_chain"]
    assert roles.numel() == speakers.numel() == chains.numel() > 0
    assert roles.dtype == speakers.dtype == chains.dtype == torch.long
    assert set(roles.tolist()) <= {ROLE_USER, ROLE_BYSTANDER}
    assert int(speakers.min()) >= 0  # every turn on a session row has a talker
    # One chain draw per row: one chain id, and never the pad value.
    assert len(set(chains.tolist())) == 1
    assert int(chains[0]) > 0


def _dilate(mask: np.ndarray, frames: int) -> np.ndarray:
    """Grow a frame mask both ways.

    The turn gate is a raised cosine ``fade_samples`` wide, so the signal opens
    slightly before its turn's first frame and closes after its last; the label
    is the script's span, not the taper's. Two or three frames of slack is that
    taper, not a misalignment.
    """
    out = mask.copy()
    for shift in range(1, frames + 1):
        out[shift:] |= mask[:-shift]
        out[:-shift] |= mask[shift:]
    return out


def test_labels_stay_on_the_grid_under_speed_perturbation(corpus):
    """The script is written before the speed change and read after it.

    A script mapped with the wrong factor would put the user's activity outside
    the spans that claim it, and the error would grow along the row -- so the
    test is "nearly all of the user's active frames sit inside a user turn",
    which a 5% scaling error on a 20 s row could not satisfy.

    The threshold is the measured one, not a round number: over 24 seeded rows
    the correct mapping gives 0.998 at the worst row and 1.000 at the median,
    while dropping the speed factor (mapping the script with 1.0) gives 0.774
    at the worst row and 0.968 at the median -- so 0.99 separates the two and
    0.95 would not have caught the injected bug on half its rows.

    The two allowances are the contract, not slack. ``shared`` -- frames where
    both energy VADs fire -- must be allowed because ``turn_id`` is 0 on double
    talk BY CONTRACT; without it the same measurement reads 0.81 at the median
    on correct code, i.e. it would be testing the contract's opposite. The
    3-frame dilation is the gate: ``fade_samples`` is 400 samples (2.5 frames)
    and the labeler's analysis window is another 400, so a turn's first frame
    of audible energy can precede its first labelled frame.
    """
    dataset = VoiceIsolationDataset(
        metafile_path=str(corpus),
        min_utt_length_in_seconds=1.0,
        min_utts_in_each_speaker=1,
        target_sr=SR,
        training_sample_length_in_seconds=20.0,
        audio_gain_normalized_to=-28,
        augmentation_speech_args=_SPEECH_ARGS,
        augmentation_reverb_args=_REVERB_ARGS,
        augmentation_speed_args={"used": True, "prob": 1.0, "speed_range": [0.95, 1.05]},
        augmentation_session_rows_args=_session(),
        vad_label_args=_VAD_ARGS,
    )
    for seed in range(8):
        sample = _row(dataset, 4200 + seed, seconds=20.0)
        turn_id = sample["turn_id"].numpy()
        active = sample["user_active"].numpy() > 0
        assert len(turn_id) == len(active)
        roles = sample["turn_role"].numpy()
        user_ids = [k for k in range(1, len(roles) + 1) if roles[k - 1] == ROLE_USER]
        user_frames = np.isin(turn_id, user_ids)
        assert user_frames.any()
        # double talk carries no id, so the user's own turns plus the frames a
        # bystander shares with them are the whole of "the user is speaking"
        shared = active & (sample["bystander_active"].numpy() > 0)
        allowed = _dilate(user_frames | shared, 3)
        inside = float((active & allowed).sum()) / float(active.sum())
        assert inside >= 0.99, (seed, inside)


# --------------------------------------------------------------------------- #
# 3./4. the gap: present in the target's absence, never digital silence
# --------------------------------------------------------------------------- #


def _gap_dataset(corpus):
    return _dataset(
        corpus,
        _session(
            min_seconds=6.0,
            shape_probs={
                "user_first": 0.0,
                "bystander_first": 0.0,
                "user_gap": 1.0,
                "overlap": 0.0,
            },
            user_gap_seconds=[5.0, 6.0],
            user_turn_seconds=[1.5, 3.0],
            bystander_in_gap_prob=0.0,
        ),
        seconds=20.0,
    )


def test_a_long_user_gap_is_never_a_target_absent_row(corpus):
    dataset = _gap_dataset(corpus)
    seen = 0
    for seed in range(8):
        sample = _row(dataset, 5000 + seed, seconds=20.0)
        assert float(sample["session_gap_seconds"]) >= 5.0
        assert float(sample["target_absent"]) == 0.0
        assert float(sample["target_present"]) == 1.0
        assert float(sample["clean_speech"].abs().max()) > 0.0
        zeros = _runs((sample["user_active"] > 0).numpy().astype(int), 0)
        longest = max(end - start for start, end in zeros)
        assert longest >= int(4.5 * FPS), (seed, longest / FPS)
        seen += 1
    assert seen == 8


def test_the_gap_carries_the_rows_floor_not_digital_silence(corpus):
    """The forced capture floor is what the model hears while the user is away.

    Noise is OFF here on purpose: this is the ~10% of rows where every noise
    source declines to fire, and it is exactly the case where a gap used to be
    digital zeros.
    """
    dataset = _gap_dataset(corpus)
    for seed in range(6):
        sample = _row(dataset, 5100 + seed, seconds=20.0)
        quiet = ((sample["user_active"] == 0) & (sample["bystander_active"] == 0)).numpy()
        # Erode by the gate taper: the frames next to a turn edge carry the
        # raised-cosine ramp, which is speech at a low level, not floor.
        quiet &= ~_dilate(~quiet, 4)
        assert quiet.any()
        frames = np.flatnonzero(quiet)
        noisy = sample["noisy_speech"].reshape(-1)
        clean = sample["clean_speech"].reshape(-1)
        starts = frames * HOP
        window = np.concatenate([np.arange(s, min(s + HOP, noisy.shape[-1])) for s in starts])
        gap_noisy = noisy[window]
        gap_clean = clean[window]
        # the mixture is never digital silence there ...
        assert float(gap_noisy.abs().min()) > 0.0
        assert float(gap_noisy.abs().max()) > 0.0
        # ... it sits at the level the row drew, to a quarter of a dB. The band
        # is the measured one: with the 4-frame erosion above, |measured -
        # drawn| is 0.037 dB at the worst of 16 seeded rows and 0.002 dB at the
        # median (without the erosion the taper's speech pushes it to 1.31 dB,
        # which is what the erosion is for). A band of 3 dB would have accepted
        # a floor mixed at the wrong level, or a noise source leaking in.
        drawn = float(sample["session_floor_dbfs"])
        measured = 20.0 * np.log10(float(gap_noisy.pow(2).mean().sqrt()) + 1e-20)
        assert drawn - 0.25 < measured < drawn + 0.25, (seed, measured, drawn)
        # ... and the target is silent, because the user is not talking
        speaking = float(clean.pow(2).mean().sqrt())
        assert float(gap_clean.pow(2).mean().sqrt()) < 1e-3 * speaking


# --------------------------------------------------------------------------- #
# 5. one chain draw, applied jointly
# --------------------------------------------------------------------------- #


def test_the_device_chain_fires_exactly_once_per_session_row(corpus):
    dataset = _dataset(corpus, _session())
    calls = []
    original = dataset.device_chain.apply

    def spy(noisy, target, *, sample_rate):
        calls.append((noisy.shape, target.shape))
        return original(noisy, target, sample_rate=sample_rate)

    dataset.device_chain.apply = spy
    sample = _row(dataset, 6000)
    assert float(sample["session_row"]) == 1.0
    assert len(calls) == 1
    # the pair handed to the chain is the row's mixture and its target
    assert calls[0][0] == calls[0][1]


def test_noisy_minus_target_is_the_bystanders_plus_the_floor(corpus):
    """With the chain and the noise stage off, superposition is exact.

    A per-talker chain draw -- the thing review item 6 forbids -- would break
    this identity, because the bystander bus in the mixture would no longer be
    the bystander bus the row reports.

    ``rel`` is 0.02 because the residual really is exact: what is left after
    subtracting the reported bystander bus is the floor and only the floor, and
    the only slack is the sample RMS of a finite draw of Gaussian noise.
    Measured over 16 seeded rows: |measured/amplitude - 1| <= 0.0044.
    """
    dataset = _dataset(corpus, _session(), reverb=False, seconds=15.0)
    for seed in range(6):
        sample = _row(dataset, 7000 + seed)
        assert float(sample["session_row"]) == 1.0
        assert float(sample["noisy_speech"].abs().max()) < 1.0  # no rescale
        residual = (sample["noisy_speech"] - sample["clean_speech"]).reshape(-1)
        far = sample["far_target"].reshape(-1)[: residual.shape[0]]
        floor = residual - far
        amplitude = 10.0 ** (float(sample["session_floor_dbfs"]) / 20.0)
        measured = float(floor.pow(2).mean().sqrt())
        assert measured == pytest.approx(amplitude, rel=0.02), (seed, measured, amplitude)


# --------------------------------------------------------------------------- #
# 6. collate
# --------------------------------------------------------------------------- #


def test_collate_pads_the_labels_across_a_mixed_batch(corpus):
    """A batch of a session row and a plain row: K_max comes from the session
    row, T from the longest signal, and the pad values are the documented ones.
    """
    dataset = _dataset(corpus, _session(min_seconds=12.0), seconds=15.0)
    session_row = _row(dataset, 8000, seconds=15.0)
    plain_row = _row(dataset, 8001, seconds=6.0)
    batch = VoiceIsolationCollateFunc()([session_row, plain_row])

    turns = int(session_row["turn_role"].numel())
    assert batch["turn_role"].shape == (2, turns)
    assert batch["turn_speaker"].shape == (2, turns)
    assert batch["turn_chain"].shape == (2, turns)
    assert torch.equal(batch["turn_role"][1], torch.zeros(turns, dtype=torch.long))
    assert torch.equal(batch["turn_speaker"][1], torch.full((turns,), -1, dtype=torch.long))
    assert batch["row_source_id"].shape == (2,)
    assert batch["row_source_id"].dtype == torch.long
    frames = batch["user_active"].shape[-1]
    assert batch["turn_id"].shape == (2, frames)
    assert batch["bystander_active"].shape == (2, frames)
    assert float(batch["session_row"][0]) == 1.0
    assert float(batch["session_row"][1]) == 0.0


def test_collate_is_untouched_when_no_row_carries_labels(corpus):
    dataset = _dataset(corpus, None)
    batch = VoiceIsolationCollateFunc()([_row(dataset, 8100), _row(dataset, 8101)])
    for key in ("user_active", "turn_id", "turn_role", "row_source_id", "session_row"):
        assert key not in batch


def test_collate_of_a_batch_with_no_session_row_gives_empty_turn_tensors(corpus):
    dataset = _dataset(corpus, _session(prob=1.0, min_seconds=30.0))
    batch = VoiceIsolationCollateFunc()(
        [_row(dataset, 8200, seconds=6.0), _row(dataset, 8201, seconds=6.0)]
    )
    assert batch["turn_role"].shape == (2, 0)
    assert batch["turn_id"].abs().sum() == 0
    assert torch.equal(batch["row_source_id"], torch.tensor([-1, -1]))


def test_collate_session_labels_needs_every_row_to_carry_a_key():
    """A partially present key would produce a tensor shorter than the batch;
    the helper drops it instead."""
    complete = {
        "user_active": torch.ones(5),
        "bystander_active": torch.zeros(5),
        "turn_id": torch.zeros(5, dtype=torch.long),
        "turn_role": torch.ones(2, dtype=torch.long),
        "turn_speaker": torch.ones(2, dtype=torch.long),
        "turn_chain": torch.ones(2, dtype=torch.long),
        "row_source_id": torch.tensor(3),
    }
    partial = {"row_source_id": torch.tensor(-1)}
    out = collate_session_labels([complete, partial], {})
    assert "user_active" not in out
    assert out["row_source_id"].tolist() == [3, -1]


# --------------------------------------------------------------------------- #
# 7. the move
# --------------------------------------------------------------------------- #


def test_a_moved_user_is_still_the_same_speaker(corpus):
    dataset = _dataset(
        corpus, _session(rir_move_prob=1.0, user_turn_seconds=[1.5, 2.5]), seconds=20.0
    )
    moved = 0
    for seed in range(8):
        sample = _row(dataset, 9000 + seed, seconds=20.0)
        roles = sample["turn_role"].tolist()
        speakers = sample["turn_speaker"].tolist()
        user_ids = {s for role, s in zip(roles, speakers) if role == ROLE_USER}
        assert len(user_ids) == 1, (seed, roles, speakers)
        if float(sample["session_rir_move"]) == 1.0:
            moved += 1
            # two seats, and they are not the same seat
            assert float(sample["session_move_distance_delta"]) > 0.0
            # bystanders are somebody else
            bystanders = {s for role, s in zip(roles, speakers) if role == ROLE_BYSTANDER}
            assert not (bystanders & user_ids)
    assert moved >= 6, moved


def test_a_distance_matched_bystander_lands_in_the_users_range(corpus):
    dataset = _dataset(
        corpus,
        _session(distance_matched_bystander_prob=1.0, n_bystanders=[1, 1]),
        seconds=15.0,
    )
    matched = 0
    for seed in range(8):
        sample = _row(dataset, 9100 + seed)
        if float(sample["session_matched_bystander"]) != 1.0:
            continue
        matched += 1
        assert float(sample["nearest_interferer_distance"]) <= 1.0
    assert matched >= 6, matched


# --------------------------------------------------------------------------- #
# the script, on its own
# --------------------------------------------------------------------------- #


def test_the_script_always_gives_the_user_and_a_bystander_a_turn():
    config = SessionRowsConfig(enabled=True, prob=1.0, min_seconds=6.0)
    torch.manual_seed(0)
    for _ in range(200):
        for n_frames in (600, 1200, 3000):
            script = build_session_script(config, n_frames, FPS, 2)
            assert any(turn.role == ROLE_USER for turn in script.turns)
            assert any(turn.role == ROLE_BYSTANDER for turn in script.turns)
            assert script.turns == sorted(script.turns, key=lambda t: (t.start, t.end))
            for turn in script.turns:
                assert 0 <= turn.start < turn.end <= n_frames


def test_two_bystanders_never_talk_at_once():
    """The overlap clamp is what keeps `turn_id` contiguous: a turn may lose its
    edges to double talk, never its middle."""
    config = SessionRowsConfig(
        enabled=True, prob=1.0, min_seconds=6.0, boundary_overlap_prob=1.0,
        overlap_seconds=(0.5, 3.0),
    )
    torch.manual_seed(1)
    for _ in range(300):
        script = build_session_script(config, 3000, FPS, 2)
        _, bystanders = script_masks(script, 2)
        if len(bystanders) == 2:
            assert not bool((bystanders[0] & bystanders[1]).any())


def test_every_shape_is_reachable_and_reported():
    config = SessionRowsConfig(enabled=True, prob=1.0, min_seconds=6.0)
    torch.manual_seed(2)
    shapes = {build_session_script(config, 3000, FPS, 2).shape for _ in range(400)}
    assert shapes == {"user_first", "bystander_first", "user_gap", "overlap"}


def test_turn_id_is_zero_on_exactly_the_scripted_double_talk():
    """The contract, asserted where the script is visible.

    The row-level test above can only see the energy VAD's both-active frames,
    which are a superset of the script's double talk. Here the script IS the
    reference, so the three properties a turn-pooling loss depends on are exact
    rather than statistical: no scripted double-talk frame carries an id, every
    id occupies one contiguous run, and every id lies inside its own turn's
    span (`pool_turn_means` indexes ``turn_role``/``turn_speaker`` by id, so an
    id outside its span would pool one turn's frames under another's label).
    """
    from puresound.task.session_rows import SessionRowBuilder

    config = SessionRowsConfig(
        enabled=True, prob=1.0, min_seconds=6.0, boundary_overlap_prob=1.0,
        overlap_seconds=(0.5, 3.0),
    )
    builder = SessionRowBuilder(config)
    torch.manual_seed(5)
    for _ in range(120):
        for n_frames in (600, 1200, 3000):
            script = build_session_script(config, n_frames, FPS, 2)
            turn_id, _ = builder._turn_frames(script, n_frames, 1.0)
            ids = turn_id.numpy()
            occupancy = np.zeros(n_frames, dtype=int)
            for turn in script.turns:
                occupancy[turn.start : min(turn.end, n_frames)] += 1
            assert not (ids[occupancy > 1] != 0).any()
            for k in range(1, int(ids.max()) + 1):
                frames = np.flatnonzero(ids == k)
                if not frames.size:
                    continue
                assert frames[-1] - frames[0] + 1 == frames.size, (k, frames)
                turn = script.turns[k - 1]
                assert frames[0] >= turn.start
                assert frames[-1] < min(turn.end, n_frames)


def test_a_shape_the_row_cannot_hold_is_demoted_not_truncated():
    config = SessionRowsConfig(
        enabled=True, prob=1.0, min_seconds=1.0, user_gap_seconds=(5.0, 20.0),
        shape_probs={"user_first": 0.0, "bystander_first": 0.0,
                     "user_gap": 1.0, "overlap": 0.0},
    )
    torch.manual_seed(3)
    for _ in range(50):
        script = build_session_script(config, 200, FPS, 1)  # 2 s of row
        assert script.shape != "user_gap"
        assert script.gap_seconds == 0.0


# --------------------------------------------------------------------------- #
# pairing: same source material, independent chain
# --------------------------------------------------------------------------- #


def test_the_source_scope_restores_every_stream_it_seeded():
    torch.manual_seed(11)
    import random as _random

    _random.seed(11)
    np.random.seed(11)
    expected = (torch.rand(3), _random.random(), np.random.rand(2))

    torch.manual_seed(11)
    _random.seed(11)
    np.random.seed(11)
    with source_rng_scope(999):
        torch.rand(17)
        _random.random()
        np.random.rand(5)
    got = (torch.rand(3), _random.random(), np.random.rand(2))

    assert torch.equal(expected[0], got[0])
    assert expected[1] == got[1]
    assert np.allclose(expected[2], got[2])


def test_source_scope_is_a_noop_without_a_seed():
    torch.manual_seed(12)
    expected = torch.rand(4)
    torch.manual_seed(12)
    with source_rng_scope(None):
        pass
    assert torch.equal(expected, torch.rand(4))


def test_paired_rows_share_their_material_and_not_their_chain(corpus):
    """Two rows with the same ``row_source_id`` are the same session through two
    independent chain draws -- the pairs a cross-chain consistency term reads.
    """
    dataset = _dataset(
        corpus,
        _session(pair_prob=1.0, pair_pool_size=1, min_seconds=6.0),
        seconds=15.0,
    )
    rows = [_row(dataset, 10000 + seed) for seed in range(4)]
    ids = {int(row["row_source_id"]) for row in rows}
    assert len(ids) == 1 and ids != {-1}
    reference = rows[0]
    for row in rows[1:]:
        # same script, same talkers, same level
        assert torch.equal(row["turn_role"], reference["turn_role"])
        assert torch.equal(row["turn_speaker"], reference["turn_speaker"])
        assert float(row["session_sir_db"]) == float(reference["session_sir_db"])
        assert torch.equal(row["turn_id"], reference["turn_id"])
        # ...and a chain identity of its own
        assert int(row["turn_chain"][0]) != int(reference["turn_chain"][0])
    # an unpaired recipe reports the default
    unpaired = _dataset(corpus, _session(pair_prob=0.0), seconds=15.0)
    assert int(_row(unpaired, 10100)["row_source_id"]) == -1


def test_the_pair_id_separates_the_length_buckets(corpus):
    """The same slot in a 12 s and a 20 s batch is different material, and the
    id has to say so or a pair would compare two different sessions."""
    dataset = _dataset(
        corpus, _session(pair_prob=1.0, pair_pool_size=1, min_seconds=6.0), seconds=20.0
    )
    short = int(_row(dataset, 10200, seconds=12.0)["row_source_id"])
    long_row = int(_row(dataset, 10201, seconds=20.0)["row_source_id"])
    assert short != long_row
