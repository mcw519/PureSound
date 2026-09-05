import logging
import random
from collections import defaultdict
from copy import deepcopy
from typing import List, Mapping, NamedTuple, Optional, TypeVar, Union

import numpy as np
import torch
from pydantic import BaseModel

from puresound.audio.augmentation import AudioEffectAugmentor
from puresound.config import delegated_kwargs, with_overrides
from puresound.config.curriculum import CurriculumConfig
from puresound.config.augmentation import (
    ContinuousSpeedAugmentation,
    HighPassAugmentation,
    NoiseAugmentation,
    ReverbAugmentation,
    SimpleProbAugmentation,
    SourceRateAugmentation,
    SpeechAugmentation,
    VadLabelConfig,
    VolumeAugmentation,
    CompressorAugmentation,
)
from puresound.audio.io import AudioIO
from puresound.audio.noise import add_bg_white_noise
from puresound.audio.vad import EnergyVADLabeler, create_vad_labeler, frame_count
from puresound.dataset.parser import MetafileParser


logger = logging.getLogger(__name__)


#: A dataset argument may arrive as its validated model or as a plain mapping.
AugmentationArg = Union[BaseModel, Mapping, None]
BlockModel = TypeVar("BlockModel", bound=BaseModel)


def as_block(value: AugmentationArg, model: type[BlockModel]) -> BlockModel | None:
    """Accept either a validated config block or a mapping to validate into one.

    A recipe always hands over models. Tests and one-off scripts build the
    blocks inline, and routing those through the same model is what makes the
    schema the single gate: constructing a dataset directly no longer skips the
    validation a recipe gets.
    """
    if value is None:
        return value
    if isinstance(value, model):
        return value
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="python")
    return model.model_validate(value)


class ItemKey(NamedTuple):
    """One row's identity, as a sampler describes it.

    Everything past the speaker and its sample rate is optional and is present
    only when the run asked for it: a seed makes the row reproducible, a length
    comes from a mixed-length schedule, an epoch from a curriculum.
    """

    speaker: str
    sample_rate: Optional[int]
    seed: Optional[int] = None
    seconds: Optional[float] = None
    epoch: Optional[int] = None


class ForegroundReverb(NamedTuple):
    """Foreground after its room channel, with the reference it is scored against."""

    noisy: torch.Tensor
    clean: torch.Tensor
    metadata: Optional[dict]


class ReverbedSource(NamedTuple):
    """One non-foreground source after its room channel, with that channel's metadata."""

    wav: torch.Tensor
    metadata: Optional[dict]


class DynamicBaseDataset(torch.utils.data.Dataset):
    #: Which augmentation blocks this dataset accepts, and the model that
    #: validates each: ``keyword name -> config model``.
    #:
    #: A registry rather than a parameter list because the recipe side already
    #: derives its half from the schema's own fields, and a hand-written list
    #: facing a derived one drifts. It did: `vad_label` is on `BaseRecipe`, so
    #: `augmentation_kwargs()` emitted `vad_label_args` for every task, and
    #: `SpeakerEmbeddingDataset` -- which spelled its eight blocks out and did
    #: not list that one -- raised `TypeError` on every speaker-embedding recipe
    #: in the tree. Both halves are derived now.
    #:
    #: Subclasses extend it, and may replace an entry to bind a different
    #: dialect of the same block (speaker embedding takes discrete speeds where
    #: the separation tasks take a continuous range).
    AUGMENTATION_BLOCKS: Mapping[str, type] = {
        "augmentation_speech_args": SpeechAugmentation,
        "augmentation_noise_args": NoiseAugmentation,
        "augmentation_reverb_args": ReverbAugmentation,
        "augmentation_speed_args": ContinuousSpeedAugmentation,
        "augmentation_ir_response_args": SimpleProbAugmentation,
        "augmentation_src_args": SourceRateAugmentation,
        "augmentation_hpf_args": HighPassAugmentation,
        "augmentation_volume_args": VolumeAugmentation,
        "augmentation_compressor_args": CompressorAugmentation,
        "vad_label_args": VadLabelConfig,
    }

    def __init__(
        self,
        metafile_path: str,
        min_utt_length_in_seconds: float = 3.0,
        min_utts_in_each_speaker: int = 5,
        target_sr: Optional[int] = None,
        training_sample_length_in_seconds: float = 6.0,
        audio_gain_normalized_to: Optional[int] = None,
        dataset_role: str = "train",
        pipeline_role: str | None = None,
        curriculum: AugmentationArg = None,
        **augmentation: AugmentationArg,
    ):
        super().__init__()
        # Matafile related
        self.metafile_path = metafile_path
        self.min_utt_length_in_seconds = min_utt_length_in_seconds
        self.min_utts_in_each_speaker = min_utts_in_each_speaker

        # Audio related
        self.target_sr = target_sr
        self.audio_gain_normalized_to = audio_gain_normalized_to
        self.training_sample_length_in_seconds = training_sample_length_in_seconds
        if self.target_sr is not None:
            self.training_sample_length = int(
                self.target_sr * self.training_sample_length_in_seconds
            )
        else:
            self.training_sample_length = None

        # Augmentation related. Every block in the registry gets an attribute
        # whether the caller supplied it or not -- read sites say
        # `if self.augmentation_noise_args:` and a missing attribute would be an
        # AttributeError where None is the answer.
        blocks = type(self).AUGMENTATION_BLOCKS
        unexpected = sorted(set(augmentation) - set(blocks))
        if unexpected:
            raise TypeError(
                f"{type(self).__name__}() got unexpected keyword argument(s) "
                f"{unexpected}; it accepts {sorted(blocks)}"
            )
        for name, model in blocks.items():
            setattr(self, name, as_block(augmentation.get(name), model))
        self.dataset_role = str(dataset_role)
        if self.dataset_role not in {"train", "validation", "test"}:
            raise ValueError(
                "dataset_role must be train, validation, or test"
            )
        self.pipeline_role = (
            self.dataset_role if pipeline_role is None else str(pipeline_role)
        )
        if self.pipeline_role not in {"train", "validation", "test"}:
            raise ValueError("pipeline_role must be train, validation, or test")
        self.vad_labeler = None
        # Knobs that move with the epoch. The epoch itself arrives per item from
        # the sampler, because this object is a copy living in a worker process.
        self.curriculum = as_block(curriculum, CurriculumConfig)
        self._curriculum_epoch: int | None = None

        self.init_necessary()

    def init_necessary(self):
        self.meta, self.gender_meta, self.gender_spks, self.sr_meta = self.gen_meta(
            metafile_path=self.metafile_path,
            min_utt_length=self.min_utt_length_in_seconds,
            min_utts_in_spk=self.min_utts_in_each_speaker,
        )

        self.total_spks = sorted(list(self.meta.keys()))
        # add speaker index
        self.spk2idx = {}
        for idx, spkid in enumerate(self.total_spks):
            self.spk2idx[spkid] = idx

        self.init_augmentor()
        self.init_vad_labeler()
        # Epoch 0's values, so a row drawn before any epoch is announced (a
        # dump, an audit script, a plain PyTorch loop) still sees the schedule's
        # starting point rather than the file's constants.
        self.apply_curriculum_epoch(0)

    def apply_curriculum_epoch(self, epoch: int) -> None:
        """Move this epoch's scheduled knobs into place.

        Called per item from ``__getitem__`` with the epoch the sampler attached,
        and a no-op once that epoch is already in force -- the work is rebuilding
        a few small config objects, not per-row cost. It must never draw from the
        RNG: rows are seeded per item, and a draw here would shift every
        subsequent draw in the row depending on which epoch it happened to be.

        A worker warns rather than raises on a target it cannot apply. The
        recipe already refused an unknown target at load time
        (`BaseRecipe.curriculum_targets_exist`), so anything reaching here is a
        dataset built by hand; killing a DataLoader worker mid-epoch would hang
        DDP on the next all-reduce, which is a much worse way to learn about it.
        """
        curriculum = getattr(self, "curriculum", None)
        if curriculum is None or not curriculum.used:
            return
        epoch = int(epoch)
        if epoch == self._curriculum_epoch:
            return

        values = curriculum.resolve(epoch)
        overrides = values.augmentation_overrides()
        for block_name, fields in overrides.items():
            attribute = f"{block_name}_args"
            block = getattr(self, attribute, None)
            if block is None:
                logger.warning(
                    "curriculum schedules %s, which this dataset does not have "
                    "(disabled block?) -- skipped", block_name,
                )
                continue
            setattr(self, attribute, with_overrides(block, **fields))
        if overrides:
            self.rebind_augmentation_blocks()

        if values.bank_weights:
            room_bank = getattr(self.augmentor, "room_bank", None)
            setter = getattr(room_bank, "set_weights", None)
            if setter is None:
                logger.warning(
                    "curriculum schedules bank weights but the room bank is not a "
                    "union of banks -- skipped",
                )
            else:
                setter(values.bank_weights)

        self._curriculum_epoch = epoch

    def rebind_augmentation_blocks(self) -> None:
        """Re-derive whatever was composed from an augmentation block.

        Blocks are read per row, but the components built *from* them -- a
        capture chain, a noise stage, a gating helper -- are composed once and
        keep a reference to the block they were given. A dataset that composes
        any of them re-composes them here, so that a value which changes while
        the run is going (a curriculum) reaches them too instead of being read
        from a stale copy.

        Compose only: this runs between rows, so it must not draw from the RNG
        stream or load anything from disk. The base has nothing to re-derive.
        """

    def parse_item_key(self, key) -> "ItemKey":
        """Normalise a sampler's item key and apply what every task shares.

        Samplers hand a dataset a tuple that grows with what the run asked for:
        ``(speaker, sample_rate)``, plus a per-item seed, plus this batch's row
        length, plus this epoch (see ``puresound.task.sampler``). Parsing that in
        one place is what lets a dataset gain those capabilities by calling this
        rather than by growing its own ladder of tuple arities -- and keeps the
        two shared side effects in one order:

        1. the row length for this item, which ``sample_length`` reads,
        2. this epoch's scheduled knobs, applied before
        3. the per-item seed, so that every draw the seed governs already sees
           the values the epoch asked for.
        """
        if not isinstance(key, (tuple, list)) or not 2 <= len(key) <= 5:
            raise TypeError(
                "an item key is (speaker, sample_rate[, seed[, seconds[, epoch]]]), "
                f"got {key!r}"
            )
        speaker, sample_rate, *rest = key
        seed, seconds, epoch = (list(rest) + [None, None, None])[:3]

        self._row_length_override = (
            int(self.audio_sr * float(seconds)) if seconds is not None else None
        )
        if epoch is not None:
            self.apply_curriculum_epoch(epoch)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed % (2**32))
            torch.manual_seed(seed)
        return ItemKey(
            speaker=speaker,
            sample_rate=int(sample_rate) if sample_rate is not None else None,
            seed=seed,
            seconds=seconds,
            epoch=epoch,
        )

    def init_vad_labeler(self):
        cfg = self.vad_label_args
        # Coarse activity labeler used only by overlap gating; it runs inside
        # the DataLoader workers so it must stay cheap (energy VAD), never
        # Silero.
        self.gating_vad_labeler = None
        # When the loss label uses Silero, defer it to a batched GPU pass owned
        # by the training module: __getitem__ emits the clean reference
        # waveform and the module labels the whole batch at once on GPU,
        # lifting Silero out of the per-sample worker path entirely.
        self.defer_vad_to_gpu = False

        if cfg is None or not cfg.used:
            self.vad_labeler = None
            return

        # `args` is the labeler constructor's own namespace and wins over the
        # block-level shorthand, as it always has.
        self.gating_vad_labeler = EnergyVADLabeler(
            frame_length=cfg.args.get("frame_length", cfg.frame_length),
            hop_length=cfg.args.get("hop_length", cfg.hop_length),
            eps_mode=cfg.args.get("eps_mode", "absolute"),
        )

        if cfg.backend == "silero":
            self.defer_vad_to_gpu = True
            self.vad_labeler = None
        else:
            self.vad_labeler = create_vad_labeler(delegated_kwargs(cfg))

    # ------------------------------------------------------------------ #
    # Derived audio parameters.
    #
    # Both fall back to the *incoming* file's rate, which ``__getitem__``
    # records on ``self.ori_audio_sr`` as it opens the foreground utterance, so
    # they are only meaningful inside item synthesis -- exactly the precondition
    # the inline ``if_none_else(...)`` expressions they replace already carried.
    # The two agree on when the fallback applies: ``training_sample_length`` is
    # None if and only if ``target_sr`` is None (see __init__).
    # ------------------------------------------------------------------ #
    @property
    def audio_sr(self) -> int:
        """Sample rate every waveform on the synthesis path shares."""
        return self.target_sr if self.target_sr is not None else self.ori_audio_sr

    @property
    def sample_length(self) -> int:
        """Training crop length in samples, at ``audio_sr``.

        A length-schedule batch overrides it per item (set at the top of
        ``__getitem__``, cleared on every item so a stale value cannot leak into
        the next one; each DataLoader worker owns its own dataset copy and
        synthesises one item at a time, so the attribute is not shared).
        """
        override = getattr(self, "_row_length_override", None)
        if override is not None:
            return int(override)
        if self.training_sample_length is not None:
            return self.training_sample_length
        return int(self.ori_audio_sr * self.training_sample_length_in_seconds)

    def __len__(self):
        """Dynamic synthesis has no fixed epoch size.

        Iteration is driven by a ``batch_sampler`` (see ``task.sampler``), which
        carries its own length, so nothing on the training path asks the dataset
        for one. Raising is the honest answer -- the previous ``pass`` returned
        ``None``, which any caller would have hit as a bare ``TypeError``.
        Lightning's ``sized_len`` treats both the same (it catches
        ``NotImplementedError``), so this is not a behaviour change for it.
        """
        raise NotImplementedError

    def __getitem__(self):
        raise NotImplementedError

    def gen_meta(
        self, metafile_path: str, min_utts_in_spk: int = 10, min_utt_length: float = 3.0
    ):
        """
        Generating the metadata, and also removed speaker whos utterances number less than min_utts_in_spk
        """
        logger.info("----" * 30)
        logger.info("Dataset creating and filtering")

        meta = MetafileParser.read_from_metafile(
            f_path=metafile_path, use_speaker_as_key=True
        )

        remove_key_by_length = 0
        for spk in sorted(meta.keys()):
            for utt in list(meta[spk]["utts"].keys()):
                _s = float(meta[spk]["utts"][utt]["length"]) / float(
                    meta[spk]["utts"][utt]["sr"]
                )
                if _s < float(min_utt_length):
                    del meta[spk]["utts"][utt]
                    remove_key_by_length += 1

        logger.info(
            f"remove {remove_key_by_length} utts which utts duration less than {min_utt_length} seconds."
        )

        delete_spk = []
        for spk in sorted(meta.keys()):
            if len(meta[spk]["utts"]) < min_utts_in_spk:
                delete_spk.append(spk)

        for spk in delete_spk:
            del meta[spk]

        logger.info(
            f"Delete {len(delete_spk)} speakers which less than {min_utts_in_spk} utterances."
        )

        # Add corpus id in meta file
        # We assume the speaker named as {corpus_name}_{speaker_name}
        all_corpus_id = set()
        for spk in sorted(meta.keys()):
            corpus_id = spk.strip().split("_")[0]
            meta[spk]["corpus_id"] = corpus_id
            all_corpus_id.add(corpus_id)

        # Generate meta and let gender as key
        gender_meta = {"m": {}, "f": {}, "other": {}}
        gender_spks = {"m": [], "f": [], "other": []}
        for cid in all_corpus_id:
            gender_meta["m"][cid] = []
            gender_meta["f"][cid] = []
            gender_meta["other"][cid] = []

        missed_gender_info = 0
        for spk in meta.keys():
            cid = meta[spk]["corpus_id"]
            if (
                meta[spk]["gender"].lower() == "m"
                or meta[spk]["gender"].lower() == "male"
            ):
                gender_meta["m"][cid].append(spk)
                gender_spks["m"].append(spk)

            elif (
                meta[spk]["gender"].lower() == "f"
                or meta[spk]["gender"].lower() == "female"
            ):
                gender_meta["f"][cid].append(spk)
                gender_spks["f"].append(spk)

            else:
                missed_gender_info += 1
                gender_meta["other"][cid].append(spk)
                gender_spks["other"].append(spk)

        for cid in all_corpus_id:
            logger.info(f"{cid:>10}:           male speakers = {len(gender_meta['m'][cid])}")
            logger.info(f"{cid:>10}:         female speakers = {len(gender_meta['f'][cid])}")
            logger.info(
                f"{cid:>10}:  unknow gender speakers = {len(gender_meta['other'][cid])}"
            )

        # Every corpus is kept. A min-speaker filter used to sit here, but its
        # condition was `len(...) < 0` -- never true -- so it has never dropped a
        # corpus in this repo's history; the branches only differed by a print.
        # Removed rather than "repaired" to the < 4 its message claimed:
        # reinstating a real filter changes the training distribution and belongs
        # in a deliberate experiment, not a cleanup.
        self.all_corpus_id = set(all_corpus_id)

        total_male_speakers = 0
        total_female_speakers = 0
        for cid in self.all_corpus_id:
            total_male_speakers += len(gender_meta["m"][cid])
            total_female_speakers += len(gender_meta["f"][cid])

        logger.info(f"Overall have {len(self.all_corpus_id)} corpus, {self.all_corpus_id}")
        logger.info(f"Overall have {total_male_speakers} male speakers")
        logger.info(f"Overall have {total_female_speakers} female speakers")
        logger.info(f"Missed gender information: {missed_gender_info}")
        logger.info(f"Total speakers: {len(meta.keys())}")
        logger.info("----" * 30)

        # Generate meta and let SR as key
        sr_meta = defaultdict(lambda: defaultdict(list))
        for spk in sorted(meta.keys()):
            for utt in list(meta[spk]["utts"].keys()):
                _sr = int(meta[spk]["utts"][utt]["sr"])
                sr_meta[_sr][spk].append(utt)

        for _sr in sorted(sr_meta.keys()):
            delete_spk = []
            for _spk in sorted(sr_meta[_sr].keys()):
                if len(sr_meta[_sr][_spk]) < min_utts_in_spk:
                    delete_spk.append(_spk)

            for _spk in delete_spk:
                del sr_meta[_sr][_spk]

        for _sr in sorted(sr_meta.keys()):
            if len(sr_meta[_sr].keys()) == 0:
                del sr_meta[_sr]

        logger.info(f"Overall have {len(sr_meta.keys())} sample rate in all corpus")
        for key in sorted(sr_meta.keys()):
            logger.info(f"{key:>8}: number of speakers = {len(sr_meta[key])}")
        logger.info("----" * 30)
        return meta, gender_meta, gender_spks, sr_meta

    def init_augmentor(self):
        self.augmentor = AudioEffectAugmentor()
        if self.augmentation_noise_args is not None:
            self.augmentor.load_bg_noise_from_folder(
                self.augmentation_noise_args.noise_folder
            )
            logger.info(
                f"Augmentor finished load {len(self.augmentor.bg_noise.keys())} noises"
            )

        if self.augmentation_reverb_args is not None:
            simulator_args = self.augmentation_reverb_args.simulator
            if simulator_args is not None and simulator_args.used:
                pregenerated_args = simulator_args.pregenerated
                if pregenerated_args is not None and pregenerated_args.used:
                    pregenerated_config = delegated_kwargs(pregenerated_args)
                    configured_role = pregenerated_args.usage_role
                    if (
                        configured_role is not None
                        and str(configured_role) != self.pipeline_role
                    ):
                        raise ValueError(
                            "pregenerated RIR usage_role does not match "
                            "the pipeline_role"
                        )
                    pregenerated_config["usage_role"] = self.pipeline_role
                    self.augmentor.init_room_bank(pregenerated_config)
                    mix = getattr(self.augmentor.room_bank, "describe", None)
                    logger.info(
                        "Augmentor initialized pre-generated RIR bank "
                        f"(kind={self.augmentor.room_bank_kind}, "
                        f"items={len(self.augmentor.room_bank)}"
                        + (f", mix: {mix()}" if mix is not None else "")
                        + ")"
                    )
                else:
                    self.augmentor.init_room_simulator(
                        delegated_kwargs(simulator_args)
                    )
                    logger.info("Augmentor initialized physics-based room simulator")
            else:
                self.augmentor.load_rir_from_folder(
                    self.augmentation_reverb_args.rir_folder
                )
                logger.info(f"Augmentor finished load {len(self.augmentor.rir.keys())} rirs")

            smear_args = self.augmentation_reverb_args.direct_smear
            if smear_args is not None and smear_args.used:
                self.augmentor.init_direct_smear(delegated_kwargs(smear_args))
                k = self.augmentor.direct_smear
                logger.info(
                    f"Augmentor direct-arrival smear: prob={k['prob']}, "
                    f"{k['smear_ms_range'][0]}-{k['smear_ms_range'][1]} ms"
                )

            drr_contrast_args = self.augmentation_reverb_args.drr_contrast
            if drr_contrast_args is not None and drr_contrast_args.used:
                self.augmentor.init_drr_contrast(delegated_kwargs(drr_contrast_args))
                knob = self.augmentor.drr_contrast
                if knob["mode"] == "deterministic":
                    detail = (
                        f"deterministic {knob['extra_db_per_decade']} dB/decade "
                        f"about {knob['pivot_m']} m"
                    )
                else:
                    detail = (
                        f"random prob={knob['prob']}, near +{knob['near_boost_db']} dB, "
                        f"far -{knob['far_cut_db']} dB"
                    )
                logger.info(
                    "Augmentor initialized DRR-contrast augmentation "
                    f"({detail}, window {knob['direct_window_ms']} ms)"
                )

        logger.info("----" * 30)

    def should_apply_source_level_reverb(self) -> bool:
        reverb = self.augmentation_reverb_args
        if reverb is None or not reverb.used:
            return False
        simulator = reverb.simulator
        if simulator is None or not simulator.used or not simulator.source_level:
            return False
        return (torch.rand(1) < reverb.prob).item()

    def apply_source_level_target_reverb(
        self,
        wav: torch.Tensor,
        sr: int,
        room_scene: dict,
        distance_range_override: Optional[List[float]] = None,
    ) -> "ForegroundReverb":
        """Give the foreground its channel.

        ``metadata`` carries the realized placement (e.g.
        ``source_receiver_distance``) so callers can condition on where the
        foreground actually landed; None when the RIR came from a folder instead
        of the simulator. The result is a plain 3-tuple, so existing
        ``noisy, clean, meta = ...`` unpacking is unaffected.
        """
        applied = self.augmentor.apply_rir(
            wav=wav,
            rir_mode="full",
            sr=sr,
            room_scene=room_scene,
            source_role="foreground",
            distance_range_override=distance_range_override,
        )
        target_rir_type = self.augmentation_reverb_args.target_rir_type
        if target_rir_type == "anechoic":
            clean_target = wav
        else:
            clean_target = self.augmentor.apply_rir(
                wav=wav,
                rir_id=applied.detail.rir_id,
                rir_mode=target_rir_type,
                sr=sr,
            ).wav
        return ForegroundReverb(applied.wav, clean_target, applied.detail.metadata)

    def apply_source_level_interferer_reverb(
        self,
        wav: torch.Tensor,
        sr: int,
        room_scene: dict,
        distance_range_override: Optional[List[float]] = None,
        source_role: str = "interferer",
    ) -> "ReverbedSource":
        """Give one non-foreground source its channel.

        Returns the metadata alongside the waveform rather than dropping it: the
        caller needs per-interferer placement for its RIR lineage, and the only
        other way to get it was to read the augmentor's ``_last_rir_meta``
        between calls -- a side channel that silently mis-attributes as soon as
        anything else convolves in between.
        """
        applied = self.augmentor.apply_rir(
            wav=wav,
            rir_mode="full",
            sr=sr,
            room_scene=room_scene,
            source_role=source_role,
            distance_range_override=distance_range_override,
        )
        return ReverbedSource(applied.wav, applied.detail.metadata)

    def create_vad_target(self, clean_speech: torch.Tensor, sample_rate: int):
        if self.vad_labeler is None:
            return None
        # All-zero reference (e.g. target-absent training samples) makes the
        # energy VAD return all-one due to its self-relative dB normalization;
        # short-circuit so every backend reports "no activity" consistently.
        if torch.is_tensor(clean_speech) and clean_speech.abs().max().item() == 0.0:
            return self.create_empty_vad_target(clean_speech)
        return self.vad_labeler(clean_speech, sample_rate=sample_rate)

    def create_empty_vad_target(self, wav: torch.Tensor):
        if self.vad_labeler is None:
            return None
        n_frames = frame_count(
            length=wav.shape[-1],
            frame_length=self.vad_labeler.frame_length,
            hop_length=self.vad_labeler.hop_length,
        )
        return torch.zeros(n_frames, dtype=torch.float32)

    def choose_an_utterance_by_speaker_name(
        self,
        target_speaker_name: str,
        ignoring_utt_list: Optional[List[str]] = None,
        select_channel: Optional[int] = None,
        select_with_sr_as_key: Optional[int] = None,
    ):
        """
        Random select an utterance in dataset by given a target speaker name

        Args:
            target_speaker_name: the key (name) of target speaker
            ignoring_utt_list: a list of utterance name for those we don't want to pick it
            select_channel: if given, only used the selected channel, it would make sure is single channel

        Returns:
            target speech tensor and its manifest information
        """
        timeout = 0
        if select_with_sr_as_key is None:
            target_speech_pool = deepcopy(self.meta[target_speaker_name]["utts"])
            check_key_list = list(target_speech_pool.keys())
            if ignoring_utt_list is not None:
                for ignored_key in ignoring_utt_list:
                    if ignored_key in check_key_list:
                        del target_speech_pool[ignored_key]
            target_speech_pool = list(target_speech_pool.keys())
        else:
            # Keep the pool an ordered LIST. Routing it through a set -- as this
            # branch used to -- leaves `random.sample` below indexing into a
            # str-hash-ordered sequence, so the SAME seed picks a DIFFERENT
            # utterance in every process. That silently defeats the per-item
            # seeding the seeded sampler exists for (see task/sampler.py), and
            # it is invisible in-process: you only see it by comparing two runs
            # under different PYTHONHASHSEED. Metafile order matches what the
            # `select_with_sr_as_key is None` branch above already draws from.
            #
            # Only recipes with `target_sample_rate: null` reach this branch
            # (runner.py sets select_by_sr_first from it), which is why no
            # shipped recipe was affected.
            target_speech_pool = list(
                self.sr_meta[select_with_sr_as_key][target_speaker_name]
            )
            if ignoring_utt_list is not None:
                ignored = set(ignoring_utt_list)
                target_speech_pool = [
                    key for key in target_speech_pool if key not in ignored
                ]

        # Chooce only one utterance
        tgt_key = random.sample(target_speech_pool, k=1)[0]

        target_speech, sr = AudioIO.open(
            f_path=self.meta[target_speaker_name]["utts"][tgt_key]["path"],
            target_lvl=self.audio_gain_normalized_to,
            resample_to=self.target_sr,
        )

        # Check the waveform is not empty
        while target_speech.abs().mean() == 0:
            if timeout < 5:
                timeout += 1
                logger.warning(
                    "Open an empty segment: %s.",
                    self.meta[target_speaker_name]["utts"][tgt_key]["path"],
                )
                logger.warning("Retry %d times.", timeout)
                target_speech, sr, (target_speaker_name, tgt_key) = (
                    self.choose_an_utterance_by_speaker_name(
                        target_speaker_name=target_speaker_name,
                        ignoring_utt_list=ignoring_utt_list,
                    )
                )
            else:
                raise RuntimeError("Timeout, can't find a useful utterance.")

        if select_channel is not None:
            if target_speech.shape[0] > select_channel:
                target_speech = target_speech[select_channel].reshape(1, -1)

        if select_with_sr_as_key:
            assert (
                sr == select_with_sr_as_key
            ), f"Given {select_with_sr_as_key} as target sr, but get {self.meta[target_speaker_name]['utts'][tgt_key]['path']} has {sr} sr"
        return target_speech, sr, (target_speaker_name, tgt_key)

    def apply_audio_augmentation(self):
        raise NotImplementedError

    def align_audio_list(
        self, wav_list: List[torch.Tensor], length: int, padding_type: str = "zero"
    ):
        """Randome shift, padding or truncate to match target length."""
        padding_type = padding_type.lower()
        assert padding_type in ["zero", "normal"]
        out_list = []

        for wav in wav_list:
            if wav.shape[-1] >= length:
                offset = random.randint(0, int(wav.shape[-1]) - length)
                # Avoid choosing an all-silent crop, but BOUND the search: if the
                # whole waveform is (near-)silent, no offset ever satisfies the
                # condition and this spins forever -- a stalled DataLoader worker
                # then deadlocks DDP (one rank never reaches the next collective).
                # After a few tries accept the current window; a silent segment is
                # legitimate (quiet interferer / target-absent source) and handled
                # downstream.
                silent_retry = 0
                while wav[:, offset : offset + length].abs().mean() == 0:
                    if silent_retry >= 10:
                        break
                    silent_retry += 1
                    offset = random.randint(0, int(wav.shape[-1]) - length)
                clips_wav = wav[:, offset : offset + length]
                assert clips_wav.shape[-1] == length

            else:
                ch, wav_length = wav.shape
                offset = random.randint(0, length - int(wav.shape[-1]))
                pre_padd = torch.zeros(ch, offset, device=wav.device, dtype=wav.dtype)
                suf_padd = torch.zeros(
                    ch, length - wav_length - offset, device=wav.device, dtype=wav.dtype
                )
                clips_wav = torch.cat([pre_padd, wav, suf_padd], dim=-1)
                assert clips_wav.shape[-1] == length

            if padding_type == "zero":
                pass

            elif padding_type == "normal":
                snr = [40]
                clips_wav = add_bg_white_noise(wav=clips_wav, snr_list=snr)[0]

            out_list.append(clips_wav)
        return out_list

    def avoid_audio_clipping(self, wav_list: List[torch.Tensor]):
        max_sample = np.max([x.abs().max().item() for x in wav_list])
        if max_sample > 1:
            out_list = []
            for wav in wav_list:
                out_list.append(wav / max_sample)

            return out_list
        else:
            return wav_list
