"""One user, some bystanders, and a script of who talks when.

Every other row type in this pipeline is a *scene*: a near talker and one or two
far talkers, both talking from the first frame, mixed at one level, labelled by
one waveform. That is what makes today's foreground decision cheap. Measured on
1206 rows of the shipped v16 recipe
(``benchmarks/probes/v19c_diagnostics/training_data_audit``): the target's onset
is inside 0.5 s on 90.7% of rows, an interferer holds the floor for >=1 s before
it on 6.0%, the longest target-free gap is 0.53 s at the median and >=5 s on
0.3%, and **no row renders the same talker twice**. Nothing in the objective
mentions time, and nothing in the data says which talker is "the user" beyond
"the one the row happens to call ``target``".

A *session row* is the same synthesis machinery arranged as a conversation:

* one **user** U, drawn in the recipe's near range, who takes several turns and
  may **move** between them (a second near channel of the same room);
* one or two **bystanders** B in the far range -- or, on a configurable share of
  rows, in *U's own* distance class, so proximity alone cannot say who the user
  is;
* the row's own **floor** in the gaps, never digital silence;
* per-frame and per-turn **labels** on the ``vad_target`` grid, so a loss can
  pool a turn, compare two turns of the same speaker, and know which of them was
  the user.

What this module deliberately does NOT do, each because the record forbids it:

* **It does not draw a chain per talker.** The device chain is one draw per row,
  applied jointly to the mixture and the target, exactly as
  ``NoiseSuppressionDataset.__getitem__`` does for every other row type -- a
  per-talker chain would break the mixture/target consistency every loss here
  assumes. Cross-*chain* views of one talker come from cross-*row* pairs
  (``pair_prob``), never from inside a row.
* **It does not fill a gap with digital silence.** Silence as filler is worth
  -5.90 dB (v8) / -19.22 dB (v16) of suppression bias on the lone-interferer arm
  and -2.33 dB of keep-side onset deletion; ``floor_dbfs_range`` guarantees a
  capture floor on session rows even when every noise source declines to fire.
* **It does not label a long user gap as target-absent.** Raising the
  target-absent rate cost Dawn WER 0.392 -> 0.626 and deletion 0.286. The row
  keeps ``target_present = 1`` and the separation target stays U's own signal.
* **It does not touch the length schedule.** A session is a long row inside the
  existing buckets (``min_seconds``), so rows/batch and speakers/batch are
  unchanged -- v15's budget-shortfall confound is not re-imported.

Contracts, the same three ``DeviceChain``, ``NoiseStage`` and ``OverlapGating``
hold:

**RNG order is load-bearing.** The order below is the order a session row draws
in, and moving one draw changes every row a seeded recipe produces afterwards::

    pair slot (only when pair_prob > 0)      global stream
    chain id                                 global stream
    --- source scope (seeded on a paired row, plain stream otherwise) ---
    bystander count
    turn shape, then the script's turn/gap/overlap durations
    talker ids, then one utterance per talker
    source-level-reverb draw, room scene
    user channel, the move draw, the second user channel
    matched-bystander draw, bystander channels
    SIR (low-tail regime, then the value), floor level
    --- end source scope ---

**A disabled block draws nothing.** ``enabled: False``, ``prob: 0`` and a row
shorter than ``min_seconds`` all return before the probability draw, so a recipe
with this block off -- or with it on, in its short length buckets -- regenerates
bit-identically.

**The source scope restores the streams it seeded.** A paired row renders the
material its ``row_source_id`` determines and then hands the streams back
untouched, so the noise draw and the device chain that follow are the row's own.
Two rows carrying the same ``row_source_id`` are therefore the same source
material through two independent chains, which is what a cross-chain consistency
term needs. What a twin shares: room, talkers, utterance crops, script, SIR,
floor level. What it does not: speed perturbation, the noise draw, the chain.
"""

from __future__ import annotations

import random
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from puresound.audio.noise import add_bg_noise
from puresound.audio.vad import EnergyVADLabeler, frame_count

# The same raised-cosine taper the turn-taking branch gates with, imported
# rather than reimplemented: a session turn edge and a turn-taking turn edge
# must be the same edge, or the model can hear which row type it is looking at.
from puresound.task.overlap_gating import _Envelope


#: Per-frame and per-turn labels a session row adds to the batch. This is the
#: contract the identity / proximity losses code against (v20 design §4.1);
#: names and shapes are fixed here.
#:
#: ``user_active``      float [T]      energy VAD of the dry user signal
#: ``bystander_active`` float [T]      energy VAD of the summed dry bystanders
#: ``turn_id``          long  [T]      1..K per contiguous single-talker turn,
#:                                     0 = no turn or double talk
#: ``turn_role``        long  [K_max]  1 user, 2 bystander, 0 pad
#: ``turn_speaker``     long  [K_max]  global speaker index, -1 pad
#: ``turn_chain``       long  [K_max]  this row's chain-draw id, 0 pad
#: ``row_source_id``    long  [N]      paired-source identity, -1 = unpaired
SESSION_LABEL_KEYS = (
    "user_active",
    "bystander_active",
    "turn_id",
    "turn_role",
    "turn_speaker",
    "turn_chain",
    "row_source_id",
)

#: Row-level diagnostics, emitted on EVERY row of a session-enabled recipe (the
#: scalar collate is one `cat` per key and a partially-present key would produce
#: a tensor shorter than the batch).
SESSION_SCALAR_KEYS = (
    "session_row",
    "session_shape",
    "session_n_turns",
    "session_n_user_turns",
    "session_rir_move",
    # Metres between the user's two seats, and how many DISTINCT channels the
    # provider actually handed out. A bank room with a single near channel can
    # only give the same seat back, and that is a move that did not move -- so
    # both are recorded rather than assumed.
    "session_move_distance_delta",
    "session_move_channels",
    "session_matched_bystander",
    "session_gap_seconds",
    "session_sir_db",
    "session_floor_dbfs",
)

#: Emitted as floats so they collate with the other scalars; mirrors
#: ``MIX_MODE_CODES``.
SESSION_SHAPE_CODES = {
    "none": 0.0,
    "user_first": 1.0,
    "bystander_first": 2.0,
    "user_gap": 3.0,
    "overlap": 4.0,
}

#: Roles, as emitted in ``turn_role``.
ROLE_PAD = 0
ROLE_USER = 1
ROLE_BYSTANDER = 2


# ---------------------------------------------------------------------- #
# draw helpers -- every one of them reads the shared torch/random streams,
# so the source scope below captures all of them.
# ---------------------------------------------------------------------- #


def _uniform(low: float, high: float) -> float:
    return float(torch.empty(1).uniform_(float(low), float(high)).item())


def _frames(seconds: float, fps: float) -> int:
    return max(1, int(round(float(seconds) * float(fps))))


@contextmanager
def source_rng_scope(seed: Optional[int]):
    """Render this row's *source material* from ``seed``, then step back out.

    ``None`` is a no-op, so an unpaired row draws from the shared stream like
    everything else. With a seed, the three streams the synthesis path uses are
    seeded and then **restored**, which is the whole point: the material becomes
    a function of the seed while the capture that follows stays the row's own.
    """
    if seed is None:
        yield
        return
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)


# ---------------------------------------------------------------------- #
# the script
# ---------------------------------------------------------------------- #


@dataclass
class Turn:
    """One talker holding the floor, in frames of the pre-speed grid."""

    role: int
    #: 0 = the user, 1..n = the n-th bystander.
    talker: int
    start: int
    end: int


@dataclass
class SessionScript:
    turns: List[Turn]
    shape: str
    #: The realized user gap, seconds. 0 when the row has no scripted gap.
    gap_seconds: float
    n_frames: int


def _draw_shape(weights) -> str:
    entries = [
        ("user_first", float(weights.user_first)),
        ("bystander_first", float(weights.bystander_first)),
        ("user_gap", float(weights.user_gap)),
        ("overlap", float(weights.overlap)),
    ]
    entries = [(name, w) for name, w in entries if w > 0.0]
    total = sum(w for _, w in entries)
    roll = _uniform(0.0, total)
    accumulated = 0.0
    for name, weight in entries:
        accumulated += weight
        if roll <= accumulated:
            return name
    return entries[-1][0]


def _fit_shape(shape: str, config, n_frames: int, fps: float) -> str:
    """Demote a shape the row is too short to carry.

    A 12 s row cannot hold a 5 s gap between two 3 s user turns and still open
    on an 8 s bystander; rather than silently truncate the shape into something
    else, the demotion is explicit and reported (``session_shape`` is the shape
    that was *realized*, not the one that was drawn).
    """
    minimum = _frames(config.min_turn_seconds, fps)
    if shape == "user_gap":
        need = 2 * minimum + _frames(config.user_gap_seconds[0], fps)
        if n_frames >= need:
            return shape
        shape = "bystander_first"
    if shape == "bystander_first":
        need = minimum + _frames(min(config.bystander_open_seconds), fps)
        if n_frames >= need:
            return shape
        shape = "user_first"
    return shape


def _next_start(start: int, end: int, previous_end: int, config, fps: float,
                force_overlap: bool) -> int:
    """Where the next turn begins: after a gap, or inside this turn's tail.

    The overlap is clamped so it can only reach back into the turn that is
    ending -- never past the one before it. Two bystanders never speak at once
    as a result, which is what keeps ``turn_id`` contiguous (a turn loses only
    its edges to double talk, never its middle).
    """
    if force_overlap or (
        config.boundary_overlap_prob > 0.0
        and torch.rand(1).item() < config.boundary_overlap_prob
    ):
        step = _frames(_uniform(*config.overlap_seconds), fps)
        return max(start + 1, previous_end, end - step)
    return end + _frames(_uniform(*config.turn_gap_seconds), fps)


def build_session_script(config, n_frames: int, fps: float, n_bystanders: int) -> SessionScript:
    """Draw one conversation on the frame grid.

    Returns turns in time order. Guarantees at least one user turn and at least
    one bystander turn: a session row with no user speech would be a
    target-absent row wearing a session label, and one with no bystander would
    silently skip the interferer branch.
    """
    shape = _fit_shape(_draw_shape(config.shape_probs), config, n_frames, fps)
    minimum = _frames(config.min_turn_seconds, fps)
    turns: List[Turn] = []
    gap_seconds = 0.0
    bystander_cursor = 0
    force_overlap = shape == "overlap"
    position = 0
    role = ROLE_USER
    previous_end = 0

    def _talker(next_role: int) -> int:
        nonlocal bystander_cursor
        if next_role == ROLE_USER:
            return 0
        talker = 1 + (bystander_cursor % max(1, n_bystanders))
        bystander_cursor += 1
        return talker

    if shape == "bystander_first":
        room = n_frames - minimum - _frames(config.turn_gap_seconds[0], fps)
        want = _frames(_uniform(*config.bystander_open_seconds), fps)
        opening = max(minimum, min(want, max(minimum, room)))
        turns.append(Turn(ROLE_BYSTANDER, _talker(ROLE_BYSTANDER), 0, opening))
        previous_end, position = 0, opening
        position = _next_start(0, opening, 0, config, fps, force_overlap=False)
        previous_end = opening
        role = ROLE_USER
    elif shape == "user_gap":
        head_room = n_frames - minimum - _frames(config.user_gap_seconds[0], fps)
        want = _frames(_uniform(*config.user_turn_seconds), fps)
        first = max(minimum, min(want, max(minimum, head_room)))
        turns.append(Turn(ROLE_USER, 0, 0, first))
        room = max(_frames(config.user_gap_seconds[0], fps), n_frames - first - minimum)
        gap = min(_frames(_uniform(*config.user_gap_seconds), fps), room)
        gap = max(gap, 1)
        gap_seconds = gap / fps
        # A bystander inside the gap: "the user stepped away" and "nobody is
        # near" have to stay two different conditions, and this is the row that
        # separates them.
        if n_bystanders > 0 and (
            config.bystander_in_gap_prob > 0.0
            and torch.rand(1).item() < config.bystander_in_gap_prob
        ):
            margin = _frames(0.5, fps)
            span = gap - 2 * margin
            if span >= minimum:
                length = min(_frames(_uniform(*config.bystander_turn_seconds), fps), span)
                offset = int((span - length) * torch.rand(1).item())
                start = first + margin + offset
                turns.append(
                    Turn(ROLE_BYSTANDER, _talker(ROLE_BYSTANDER), start, start + length)
                )
                previous_end = start + length
        position = first + gap
        previous_end = max(previous_end, first)
        role = ROLE_USER

    while position < n_frames and len(turns) < int(config.max_turns):
        bounds = (
            config.user_turn_seconds if role == ROLE_USER else config.bystander_turn_seconds
        )
        end = min(position + _frames(_uniform(*bounds), fps), n_frames)
        if end - position < minimum:
            break
        turns.append(Turn(role, _talker(role), position, end))
        nxt = _next_start(position, end, previous_end, config, fps, force_overlap)
        previous_end, position = end, nxt
        role = ROLE_BYSTANDER if role == ROLE_USER else ROLE_USER

    # Order matters: dropping a contained turn can remove the row's only
    # bystander, and the repair is what guarantees both roles are present.
    turns.sort(key=lambda turn: (turn.start, turn.end))
    turns = _drop_contained(turns)
    turns = _repair(turns, n_frames, minimum, n_bystanders)
    turns.sort(key=lambda turn: (turn.start, turn.end))
    return SessionScript(turns=turns, shape=shape, gap_seconds=gap_seconds,
                         n_frames=n_frames)


def _drop_contained(turns: List[Turn]) -> List[Turn]:
    """Remove a turn that sits entirely inside another. Consumes no randomness.

    A long overlap draw on a long turn can start the next turn early enough that
    it finishes before the first one does. Such a turn has no single-talker
    frame of its own -- nothing can pool it -- and it would cut the containing
    turn's exposed frames in two. Dropping it keeps the rendered audio and the
    labels the same object: the mask is built from this list.
    """
    kept: List[Turn] = []
    for index, turn in enumerate(turns):
        contained = any(
            other.start <= turn.start and turn.end <= other.end
            for position, other in enumerate(turns)
            if position != index and (other.end - other.start) > (turn.end - turn.start)
        )
        if not contained:
            kept.append(turn)
    return kept or turns


def _repair(turns: List[Turn], n_frames: int, minimum: int, n_bystanders: int) -> List[Turn]:
    """Guarantee one user turn and one bystander turn. Consumes no randomness.

    Reachable only on a row barely longer than ``min_turn_seconds``; the recipe
    keeps sessions on the long buckets, so this is a belt, not a mechanism.
    """
    if n_frames < 2:
        return turns
    if not any(turn.role == ROLE_USER for turn in turns):
        turns.append(Turn(ROLE_USER, 0, max(0, n_frames - minimum), n_frames))
    if n_bystanders > 0 and not any(turn.role == ROLE_BYSTANDER for turn in turns):
        end = min(minimum, max(1, n_frames // 2))
        turns.append(Turn(ROLE_BYSTANDER, 1, 0, end))
    return turns


def script_masks(script: SessionScript, n_bystanders: int) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Per-talker frame masks, on the script's own (pre-speed) grid."""
    n_frames = script.n_frames
    user = torch.zeros(n_frames, dtype=torch.bool)
    bystanders = [torch.zeros(n_frames, dtype=torch.bool) for _ in range(n_bystanders)]
    for turn in script.turns:
        end = min(turn.end, n_frames)
        if end <= turn.start:
            continue
        if turn.role == ROLE_USER:
            user[turn.start : end] = True
        elif 1 <= turn.talker <= n_bystanders:
            bystanders[turn.talker - 1][turn.start : end] = True
    return user, bystanders


# ---------------------------------------------------------------------- #
# the rendered row
# ---------------------------------------------------------------------- #


@dataclass
class SessionRender:
    """Everything a session row decided, carried on its ``RowPlan``."""

    script: SessionScript
    #: U through its room channel(s), gated by U's turns.
    user_full: torch.Tensor
    #: The separation target: the same, through the recipe's target RIR mode.
    user_early: torch.Tensor
    bystanders: List[torch.Tensor]
    user_metadata: Optional[dict]
    bystander_metadata: List[dict]
    user_speaker: str
    bystander_speakers: List[str]
    room_scene: Optional[dict]
    source_level_reverb: bool
    sir_db: float
    floor_dbfs: float
    chain_id: int
    row_source_id: int
    rir_move: bool
    matched_bystander: bool
    #: Which bystander index (1-based) got the user's distance class; 0 = none.
    matched_index: int
    #: The user's seat(s), metres, one per channel the row actually used.
    user_distances: List[float]
    #: The provider's own name for each of those channels (bank label / role).
    user_channels: List[str]
    hop_length: int
    frame_length: int
    #: Frames of the mask per bystander, kept for the labels.
    bystander_masks: List[torch.Tensor] = field(default_factory=list)
    user_mask: Optional[torch.Tensor] = None


class SessionRowBuilder:
    """Draws session rows and renders them; a no-op when the block is off.

    Built once per dataset. It holds no reference to the dataset -- the dataset
    is passed in per call, the way ``DeviceChain`` takes its waveforms -- so the
    two do not form a cycle that a DataLoader worker would have to collect.
    """

    def __init__(self, config, vad_labeler=None, frame_length: int = 400,
                 hop_length: int = 160):
        self.config = config
        self.frame_length = int(frame_length)
        self.hop_length = int(hop_length)
        # The label grid must be the ``vad_target`` grid. Reuse the dataset's
        # own cheap labeler when it has one (it is the same class with the same
        # frame/hop for the energy backend) so the labels cannot drift from it.
        self.labeler = vad_labeler
        if self.labeler is None or int(getattr(self.labeler, "hop_length", 0)) != self.hop_length:
            self.labeler = EnergyVADLabeler(
                frame_length=self.frame_length, hop_length=self.hop_length
            )

    # ------------------------------------------------------------------ #

    @property
    def active(self) -> bool:
        """True when the recipe declared the block enabled."""
        return bool(self.config is not None and self.config.enabled)

    def eligible(self, sample_length: int, audio_sr: int) -> bool:
        """Is this row long enough to be a session? Consumes no randomness.

        Checked before the probability draw on purpose: the short length buckets
        of a session-enabled recipe then draw exactly what they drew without the
        block, which is what makes the two comparable.
        """
        if not self.active or self.config.prob <= 0.0:
            return False
        return sample_length >= int(round(self.config.min_seconds * audio_sr))

    def maybe_build(self, dataset) -> Optional[SessionRender]:
        """The row-type draw; on a hit, render the whole session."""
        config = self.config
        if not self.eligible(dataset.sample_length, dataset.audio_sr):
            return None
        if torch.rand(1).item() >= float(config.prob):
            return None

        # Pairing and the chain identity are the two draws that must NOT come
        # from the seeded scope: the slot has to vary row by row, and the chain
        # id has to differ between two rows of the same slot.
        slot = None
        if config.pair_prob > 0.0 and torch.rand(1).item() < float(config.pair_prob):
            slot = int(torch.randint(0, int(config.pair_pool_size), (1,)).item())
        chain_id = int(torch.randint(1, 2**31, (1,)).item())

        seconds_code = min(127, int(round(dataset.sample_length / dataset.audio_sr)))
        # The bucket is part of the source identity: the same slot in a 12 s and
        # a 30 s batch are different material, and a pair must never claim
        # otherwise.
        row_source_id = -1 if slot is None else slot * 128 + seconds_code
        seed = None if slot is None else int(config.pair_seed_base) + row_source_id
        with source_rng_scope(seed):
            return self._render(dataset, chain_id=chain_id, row_source_id=row_source_id)

    # ------------------------------------------------------------------ #

    def _render(self, dataset, *, chain_id: int, row_source_id: int) -> SessionRender:
        config = self.config
        sample_rate = dataset.audio_sr
        length = dataset.sample_length
        span = min(length, int(round(float(config.max_seconds) * sample_rate)))
        fps = float(sample_rate) / float(self.hop_length)
        n_frames = frame_count(span, self.frame_length, self.hop_length)

        low, high = int(config.n_bystanders[0]), int(config.n_bystanders[1])
        n_bystanders = random.randint(low, high)
        script = build_session_script(config, n_frames, fps, n_bystanders)

        # Only the bystanders the script actually gave a turn are drawn: an
        # interferer rendered and mixed at some SIR but never audible would
        # inflate ``n_interferers`` and the row's own diagnostics.
        used = sorted({turn.talker for turn in script.turns if turn.role == ROLE_BYSTANDER})
        remap = {talker: index + 1 for index, talker in enumerate(used)}
        for turn in script.turns:
            if turn.role == ROLE_BYSTANDER:
                turn.talker = remap[turn.talker]
        n_bystanders = len(used)
        user_mask, bystander_masks = script_masks(script, n_bystanders)

        user_speaker, bystander_speakers = self._draw_talkers(dataset, n_bystanders)
        n_bystanders = len(bystander_speakers)
        bystander_masks = bystander_masks[:n_bystanders]
        user_wav = self._material(dataset, user_speaker, length)
        bystander_wavs = [self._material(dataset, spk, length) for spk in bystander_speakers]

        source_level = bool(dataset.should_apply_source_level_reverb())
        room_scene = dataset.augmentor.sample_room_scene() if source_level else None

        envelope = _Envelope(self.hop_length, int(config.fade_samples))
        (
            user_full, user_early, user_metadata, rir_move, user_distances, user_channels
        ) = self._render_user(
            dataset, user_wav, script, user_mask, room_scene, source_level,
            envelope, length, sample_rate,
        )
        bystanders, bystander_metadata, matched_index = self._render_bystanders(
            dataset, bystander_wavs, bystander_masks, room_scene, source_level,
            envelope, length, sample_rate,
        )

        sir_db = self._draw_sir(dataset)
        floor_dbfs = _uniform(*config.floor_dbfs_range)

        return SessionRender(
            script=script,
            user_full=user_full,
            user_early=user_early,
            bystanders=bystanders,
            user_metadata=user_metadata,
            bystander_metadata=bystander_metadata,
            user_speaker=user_speaker,
            bystander_speakers=bystander_speakers,
            room_scene=room_scene,
            source_level_reverb=source_level,
            sir_db=sir_db,
            floor_dbfs=floor_dbfs,
            chain_id=chain_id,
            row_source_id=row_source_id,
            rir_move=rir_move,
            matched_bystander=matched_index > 0,
            matched_index=matched_index,
            user_distances=user_distances,
            user_channels=user_channels,
            hop_length=self.hop_length,
            frame_length=self.frame_length,
            bystander_masks=bystander_masks,
            user_mask=user_mask,
        )

    # ------------------------------------------------------------------ #

    def _speaker_pool(self, dataset) -> Sequence[str]:
        """The SHARED pool -- the same one the foreground is drawn from.

        That sharing is the point: a speaker who is the user on one row is a
        bystander on another, so "user" cannot be a property of the voice in the
        training set, only of the role the row gave it.
        """
        if dataset.target_sr is not None:
            return dataset.total_spks
        return sorted(dataset.sr_meta[dataset.ori_audio_sr].keys())

    def _draw_talkers(self, dataset, n_bystanders: int) -> Tuple[str, List[str]]:
        pool = list(self._speaker_pool(dataset))
        wanted = min(n_bystanders + 1, len(pool))
        picks = random.sample(sorted(pool), k=max(1, wanted))
        return picks[0], picks[1:]

    def _material(self, dataset, speaker: str, length: int) -> torch.Tensor:
        """One utterance of this talker, cropped to the row.

        Bounded redraw on a sample-rate mismatch, for the reason
        ``_build_synthetic_interferers`` documents: an unbounded retry on a
        speaker with no utterance at this rate stalls a DataLoader worker and
        deadlocks DDP.
        """
        key = None if dataset.target_sr is not None else dataset.ori_audio_sr
        wav, sr, _ = dataset.choose_an_utterance_by_speaker_name(
            target_speaker_name=speaker, select_channel=0, select_with_sr_as_key=key,
        )
        retry = 0
        while key is not None and sr != dataset.ori_audio_sr and retry < 5:
            retry += 1
            wav, sr, _ = dataset.choose_an_utterance_by_speaker_name(
                target_speaker_name=speaker, select_channel=0, select_with_sr_as_key=key,
            )
        return dataset.align_audio_list(wav_list=[wav], length=length)[0]

    def _render_user(self, dataset, user_wav, script, user_mask, room_scene,
                     source_level, envelope, length, sample_rate):
        """U through one or two near channels of the same room.

        The move is an RIR change and nothing else: same voice, same utterance,
        same chain, a different seat in the room. With a pre-generated bank the
        second call lands on a different near channel by construction (a scene
        records the channels it has already handed out); with the on-the-fly
        simulator it is a fresh source position inside the near range.
        """
        config = self.config
        indices = [i for i, turn in enumerate(script.turns) if turn.role == ROLE_USER]
        move_at = None
        if source_level and len(indices) >= 2 and config.rir_move_prob > 0.0:
            if torch.rand(1).item() < float(config.rir_move_prob):
                pick = 1 + int((len(indices) - 1) * torch.rand(1).item())
                move_at = indices[min(pick, len(indices) - 1)]

        def _mask_for(turn_indices) -> torch.Tensor:
            mask = torch.zeros(script.n_frames, dtype=torch.bool)
            for index in turn_indices:
                turn = script.turns[index]
                mask[turn.start : min(turn.end, script.n_frames)] = True
            return mask

        if move_at is None:
            segments = [(indices, user_mask)]
        else:
            before = [i for i in indices if i < move_at]
            after = [i for i in indices if i >= move_at]
            segments = [(before, _mask_for(before)), (after, _mask_for(after))]

        full = torch.zeros(1, length)
        early = torch.zeros(1, length)
        metadata: Optional[dict] = None
        distances: List[float] = []
        channels: List[str] = []
        for turn_indices, mask in segments:
            if not turn_indices:
                continue
            if source_level:
                reverbed = dataset.apply_source_level_target_reverb(
                    wav=user_wav,
                    sr=sample_rate,
                    room_scene=room_scene,
                    distance_range_override=list(config.user_distance_range),
                )
                wet, dry, meta = reverbed.noisy, reverbed.clean, reverbed.metadata
            else:
                wet, dry, meta = user_wav, user_wav, None
            if metadata is None and meta is not None:
                metadata = dict(meta)
            if meta is not None and meta.get("source_receiver_distance") is not None:
                distances.append(float(meta["source_receiver_distance"]))
            if meta is not None:
                channels.append(
                    str(meta.get("label") or meta.get("source_role") or "")
                    + f"@{meta.get('source_receiver_distance')}"
                )
            full = full + envelope.gate(wet[..., :length], mask)
            early = early + envelope.gate(dry[..., :length], mask)
        return full, early, metadata, move_at is not None, distances, channels

    def _render_bystanders(self, dataset, wavs, masks, room_scene, source_level,
                           envelope, length, sample_rate):
        """B through the far channels -- except the matched one.

        ``distance_matched_bystander_prob`` of rows put one bystander in U's own
        distance class. Without those rows "near" answers both losses at once
        (review item 4) and a proximity scalar is a sufficient identity.
        """
        config = self.config
        matched_index = 0
        if (
            source_level
            and wavs
            and config.distance_matched_bystander_prob > 0.0
            and torch.rand(1).item() < float(config.distance_matched_bystander_prob)
        ):
            matched_index = 1 + int(torch.randint(0, len(wavs), (1,)).item())

        gated: List[torch.Tensor] = []
        metadata: List[dict] = []
        for index, wav in enumerate(wavs, start=1):
            if source_level:
                if index == matched_index:
                    reverbed = dataset.apply_source_level_interferer_reverb(
                        wav=wav,
                        sr=sample_rate,
                        room_scene=room_scene,
                        distance_range_override=list(config.user_distance_range),
                        # The near pool / near range: the role name is how both
                        # channel providers spell "this source is near".
                        source_role="foreground",
                    )
                else:
                    reverbed = dataset.apply_source_level_interferer_reverb(
                        wav=wav,
                        sr=sample_rate,
                        room_scene=room_scene,
                        distance_range_override=list(config.bystander_distance_range),
                    )
                signal, meta = reverbed.wav, reverbed.metadata
            else:
                signal, meta = wav, None
            mask = masks[index - 1] if index - 1 < len(masks) else None
            if mask is None:
                continue
            gated.append(envelope.gate(signal[..., :length], mask))
            entry = dict(meta) if meta else {}
            entry["session_role"] = "matched" if index == matched_index else "bystander"
            metadata.append(entry)
        return gated, metadata, matched_index

    def _draw_sir(self, dataset) -> float:
        """SIR for the bystander bus, with a heavier low tail than the recipe's.

        v16's realised SIR is +0.5 dB at the median with p5 -8.1, so "a bystander
        louder than the user" is a 5% condition -- while global deletion under a
        loud bystander is the failure being trained against (g ~ -5 dB at -5 dB
        SIR). The tail is a separate regime draw so its share is a knob, and it
        is floored at the recipe's own ``augmentation_speech.snr_range`` bottom:
        this block widens the tail, it never invents a level the recipe forbids.
        """
        config = self.config
        if config.sir_low_tail_prob > 0.0 and torch.rand(1).item() < float(
            config.sir_low_tail_prob
        ):
            bounds = config.sir_low_tail_range
        else:
            bounds = config.sir_range
        sir = _uniform(*bounds)
        speech = getattr(dataset, "augmentation_speech_args", None)
        floor = None
        if speech is not None and getattr(speech, "snr_range", None):
            floor = float(speech.snr_range[0])
        return sir if floor is None else max(sir, floor)

    # ------------------------------------------------------------------ #

    def mix(self, render: SessionRender, fg_wav: torch.Tensor,
            bystander_sum: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, str, float]:
        """Sum U and the bystander bus at the drawn SIR, then force a floor.

        Level mechanics are the recipe's: the same ``add_bg_noise`` every other
        row type mixes with, so the RMS normalisation and the realised-SIR
        semantics are unchanged. The floor is added afterwards, to the mixture
        only, at an absolute level -- ``augmentation_noise.absolute_floor``'s
        contract -- and it is what makes "the user is away" audibly different
        from "the stream stopped".
        """
        if float(bystander_sum.abs().amax()) == 0.0:
            noisy = fg_wav.clone()
            background = torch.zeros_like(bystander_sum)
            realized = float("nan")
        else:
            mixed, scaled = add_bg_noise(
                wav=fg_wav, noise=[bystander_sum], snr_list=[render.sir_db]
            )
            noisy, background, realized = mixed[0], scaled[0], render.sir_db
        floor = torch.randn_like(noisy) * (10.0 ** (render.floor_dbfs / 20.0))
        return noisy + floor, background, "session", realized

    # ------------------------------------------------------------------ #

    def emit_labels(self, sample: Dict, plan, *, dataset, vad_reference,
                    background_speech_reference) -> None:
        """Attach the label contract to one row -- session or not.

        Every row of a session-enabled recipe carries every key: the scalar
        collate is one ``cat`` per key, so a key present on some rows only would
        silently produce a tensor shorter than the batch. A non-session row gets
        its true per-frame activity, no turns (``turn_id`` all zero, so nothing
        pools) and ``row_source_id = -1``.
        """
        render = getattr(plan, "session", None)
        n_frames = self._grid_length(vad_reference, sample)
        user_active = self._activity(vad_reference, n_frames)
        bystander_active = self._activity(background_speech_reference, n_frames)

        if render is None:
            sample.update(
                {
                    "user_active": user_active,
                    "bystander_active": bystander_active,
                    "turn_id": torch.zeros(n_frames, dtype=torch.long),
                    "turn_role": torch.zeros(0, dtype=torch.long),
                    "turn_speaker": torch.zeros(0, dtype=torch.long),
                    "turn_chain": torch.zeros(0, dtype=torch.long),
                    "row_source_id": torch.tensor(-1, dtype=torch.long),
                }
            )
            sample.update(self._scalars(None, 0.0))
            return

        speed = float(getattr(plan, "speed_factor", 1.0)) or 1.0
        turn_id, overlap_fraction = self._turn_frames(render.script, n_frames, speed)
        roles, speakers, chains = self._turn_vectors(render, dataset)
        sample.update(
            {
                "user_active": user_active,
                "bystander_active": bystander_active,
                "turn_id": turn_id,
                "turn_role": roles,
                "turn_speaker": speakers,
                "turn_chain": chains,
                "row_source_id": torch.tensor(int(render.row_source_id), dtype=torch.long),
            }
        )
        sample.update(self._scalars(render, overlap_fraction))
        # The sampler's speaker opened an utterance this row then discarded; on a
        # paired row the user is the one the source id determined, so the row's
        # own speaker label has to follow the voice that is actually in it.
        if render.user_speaker in dataset.spk2idx:
            sample["speaker_id"] = dataset.spk2idx[render.user_speaker]
        # Realized double talk, same definition the gating branches report.
        sample["overlap_fraction"] = torch.tensor(overlap_fraction, dtype=torch.float32)

    # ------------------------------------------------------------------ #

    def _grid_length(self, vad_reference, sample: Dict) -> int:
        existing = sample.get("vad_target")
        if existing is not None:
            return int(existing.reshape(-1).shape[0])
        if vad_reference is not None:
            return frame_count(
                int(vad_reference.shape[-1]), self.frame_length, self.hop_length
            )
        return frame_count(
            int(sample["noisy_speech"].shape[-1]), self.frame_length, self.hop_length
        )

    def _activity(self, wav: Optional[torch.Tensor], n_frames: int) -> torch.Tensor:
        """Energy VAD on a dry reference, on the ``vad_target`` grid.

        All-zero short circuit, for the reason ``create_vad_target`` has one: the
        energy labeler normalises against the row's own peak, so a silent
        reference reads all-active.
        """
        if wav is None or float(wav.abs().amax()) == 0.0:
            return torch.zeros(n_frames, dtype=torch.float32)
        labels = self.labeler(wav, sample_rate=None).reshape(-1).float()
        if labels.shape[0] < n_frames:
            labels = torch.nn.functional.pad(labels, (0, n_frames - labels.shape[0]))
        return labels[:n_frames].contiguous()

    def _turn_frames(self, script: SessionScript, n_frames: int,
                     speed: float) -> Tuple[torch.Tensor, float]:
        """``turn_id`` on the post-speed grid, plus the realised overlap share.

        Speed perturbation maps sample ``i`` to ``i / speed`` and is applied
        after the script is written, so the spans are rescaled by the same
        factor; the labels then land on the grid ``vad_target`` is computed on
        (the alignment test pins them to within 2 frames).
        """
        scale = 1.0 / max(float(speed), 1e-6)
        turn_id = torch.zeros(n_frames, dtype=torch.long)
        occupancy = torch.zeros(n_frames, dtype=torch.long)
        spans: List[Tuple[int, int, int]] = []
        for index, turn in enumerate(script.turns, start=1):
            start = min(max(int(round(turn.start * scale)), 0), n_frames)
            end = min(max(int(round(turn.end * scale)), 0), n_frames)
            if end <= start:
                continue
            spans.append((index, start, end))
            occupancy[start:end] += 1
        for index, start, end in spans:
            turn_id[start:end] = index
        double_talk = occupancy > 1
        turn_id[double_talk] = 0
        # A turn keeps its LONGEST single-talker run and nothing else, so
        # ``turn_id == k`` is always one contiguous stretch whatever the script's
        # geometry -- the property a loss pooling a turn relies on. A turn with
        # no exposed frame at all simply never appears, and contributes nothing.
        for index, _, _ in spans:
            hit = (turn_id == index).nonzero().reshape(-1)
            if hit.numel() == 0:
                continue
            best_start = best_end = start = int(hit[0])
            previous = start
            for frame in hit[1:].tolist():
                if frame != previous + 1:
                    if previous - start > best_end - best_start:
                        best_start, best_end = start, previous
                    start = frame
                previous = frame
            if previous - start > best_end - best_start:
                best_start, best_end = start, previous
            turn_id[:best_start] = torch.where(
                turn_id[:best_start] == index,
                torch.zeros_like(turn_id[:best_start]),
                turn_id[:best_start],
            )
            tail = turn_id[best_end + 1 :]
            turn_id[best_end + 1 :] = torch.where(
                tail == index, torch.zeros_like(tail), tail
            )
        spoken = int((occupancy > 0).sum())
        overlap = float(int(double_talk.sum()) / spoken) if spoken else 0.0
        return turn_id, overlap

    def _turn_vectors(self, render: SessionRender, dataset):
        roles: List[int] = []
        speakers: List[int] = []
        for turn in render.script.turns:
            roles.append(int(turn.role))
            if turn.role == ROLE_USER:
                name = render.user_speaker
            else:
                index = turn.talker - 1
                name = (
                    render.bystander_speakers[index]
                    if 0 <= index < len(render.bystander_speakers)
                    else None
                )
            speakers.append(int(dataset.spk2idx.get(name, -1)) if name is not None else -1)
        chains = [int(render.chain_id)] * len(roles)
        return (
            torch.tensor(roles, dtype=torch.long),
            torch.tensor(speakers, dtype=torch.long),
            torch.tensor(chains, dtype=torch.long),
        )

    @staticmethod
    def _scalars(render: Optional[SessionRender], overlap_fraction: float) -> Dict:
        if render is None:
            values = {
                "session_row": 0.0,
                "session_shape": SESSION_SHAPE_CODES["none"],
                "session_n_turns": 0.0,
                "session_n_user_turns": 0.0,
                "session_rir_move": 0.0,
                "session_move_distance_delta": float("nan"),
                "session_move_channels": float("nan"),
                "session_matched_bystander": 0.0,
                "session_gap_seconds": 0.0,
                "session_sir_db": float("nan"),
                "session_floor_dbfs": float("nan"),
            }
        else:
            turns = render.script.turns
            seats = render.user_distances
            values = {
                "session_row": 1.0,
                "session_shape": SESSION_SHAPE_CODES.get(render.script.shape, float("nan")),
                "session_n_turns": float(len(turns)),
                "session_n_user_turns": float(
                    sum(1 for turn in turns if turn.role == ROLE_USER)
                ),
                "session_rir_move": float(render.rir_move),
                "session_move_distance_delta": (
                    abs(seats[1] - seats[0]) if len(seats) >= 2 else float("nan")
                ),
                "session_move_channels": float(len(set(render.user_channels))),
                "session_matched_bystander": float(render.matched_bystander),
                "session_gap_seconds": float(render.script.gap_seconds),
                "session_sir_db": float(render.sir_db),
                "session_floor_dbfs": float(render.floor_dbfs),
            }
        del overlap_fraction
        return {
            key: torch.tensor(value, dtype=torch.float32) for key, value in values.items()
        }


# ---------------------------------------------------------------------- #
# collate
# ---------------------------------------------------------------------- #

#: Pad value per label key. ``turn_speaker`` pads with -1 because 0 is a real
#: speaker index; ``turn_chain`` pads with 0 because a chain id is drawn from
#: [1, 2^31).
_LABEL_PADDING = {
    "user_active": 0.0,
    "bystander_active": 0.0,
    "turn_id": 0,
    "turn_role": 0,
    "turn_speaker": -1,
    "turn_chain": 0,
}


def collate_session_labels(batch: Sequence[Dict], out: Dict) -> Dict:
    """Pad the per-frame and per-turn labels into the batch.

    Nothing happens unless the rows carry them, so a recipe without the block
    collates exactly as before. ``K_max`` is the batch's longest turn list; a
    batch with no session row yields ``[N, 0]`` turn tensors, which is the
    honest shape for "no turns here" and what a loss must read as a
    zero-contribution row.
    """
    if not any("row_source_id" in item for item in batch):
        return out
    for key, padding in _LABEL_PADDING.items():
        values = [item[key].reshape(-1) for item in batch if key in item]
        if len(values) != len(batch):
            continue
        out[key] = pad_sequence(values, batch_first=True, padding_value=padding)
    ids = [item["row_source_id"].reshape(-1) for item in batch if "row_source_id" in item]
    if len(ids) == len(batch):
        out["row_source_id"] = torch.cat(ids, dim=0).long()
    return out
