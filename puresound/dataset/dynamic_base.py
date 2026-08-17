import random
from collections import defaultdict
from copy import deepcopy
from typing import Dict, List, NamedTuple, Optional

import numpy as np
import torch

from puresound.audio.augmentation import AudioEffectAugmentor
from puresound.audio.io import AudioIO
from puresound.audio.noise import add_bg_white_noise
from puresound.audio.vad import EnergyVADLabeler, create_vad_labeler, frame_count
from puresound.dataset.parser import MetafileParser


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
    def __init__(
        self,
        metafile_path: str,
        min_utt_length_in_seconds: float = 3.0,
        min_utts_in_each_speaker: int = 5,
        target_sr: Optional[int] = None,
        training_sample_length_in_seconds: float = 6.0,
        audio_gain_normalized_to: Optional[int] = None,
        augmentation_speech_args: Optional[Dict] = None,
        augmentation_noise_args: Optional[Dict] = None,
        augmentation_reverb_args: Optional[Dict] = None,
        augmentation_speed_args: Optional[Dict] = None,
        augmentation_ir_response_args: Optional[Dict] = None,
        augmentation_src_args: Optional[Dict] = None,
        augmentation_hpf_args: Optional[Dict] = None,
        augmentation_volume_args: Optional[Dict] = None,
        vad_label_args: Optional[Dict] = None,
        dataset_role: str = "train",
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

        # Augmentation related
        self.augmentation_speech_args = augmentation_speech_args
        self.augmentation_noise_args = augmentation_noise_args
        self.augmentation_reverb_args = augmentation_reverb_args
        self.augmentation_speed_args = augmentation_speed_args
        self.augmentation_ir_response_args = augmentation_ir_response_args
        self.augmentation_src_args = augmentation_src_args
        self.augmentation_hpf_args = augmentation_hpf_args
        self.augmentation_volume_args = augmentation_volume_args
        self.vad_label_args = vad_label_args
        self.dataset_role = str(dataset_role)
        if self.dataset_role not in {"train", "validation", "test"}:
            raise ValueError(
                "dataset_role must be train, validation, or test"
            )
        self.vad_labeler = None

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

        if not cfg or not cfg.get("used"):
            self.vad_labeler = None
            return

        args = dict(cfg.get("args", {}))
        frame_length = args.get("frame_length", cfg.get("frame_length", 400))
        hop_length = args.get("hop_length", cfg.get("hop_length", 160))
        self.gating_vad_labeler = EnergyVADLabeler(
            frame_length=frame_length, hop_length=hop_length
        )

        if cfg.get("backend", "energy").lower() == "silero":
            self.defer_vad_to_gpu = True
            self.vad_labeler = None
        else:
            self.vad_labeler = create_vad_labeler(cfg)

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
        """Training crop length in samples, at ``audio_sr``."""
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
        print("----" * 30)
        print("Dataset creating and filtering")

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

        print(
            f"remove {remove_key_by_length} utts which utts duration less than {min_utt_length} seconds."
        )

        delete_spk = []
        for spk in sorted(meta.keys()):
            if len(meta[spk]["utts"]) < min_utts_in_spk:
                delete_spk.append(spk)

        for spk in delete_spk:
            del meta[spk]

        print(
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
            print(f"{cid:>10}:           male speakers = {len(gender_meta['m'][cid])}")
            print(f"{cid:>10}:         female speakers = {len(gender_meta['f'][cid])}")
            print(
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

        print(f"Overall have {len(self.all_corpus_id)} corpus, {self.all_corpus_id}")
        print(f"Overall have {total_male_speakers} male speakers")
        print(f"Overall have {total_female_speakers} female speakers")
        print(f"Missed gender information: {missed_gender_info}")
        print(f"Total speakers: {len(meta.keys())}")
        print("----" * 30)

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

        print(f"Overall have {len(sr_meta.keys())} sample rate in all corpus")
        for key in sorted(sr_meta.keys()):
            print(f"{key:>8}: number of speakers = {len(sr_meta[key])}")
        print("----" * 30)
        return meta, gender_meta, gender_spks, sr_meta

    def init_augmentor(self):
        self.augmentor = AudioEffectAugmentor()
        if self.augmentation_noise_args is not None:
            self.augmentor.load_bg_noise_from_folder(
                self.augmentation_noise_args["noise_folder"]
            )
            print(
                f"Augmentor finished load {len(self.augmentor.bg_noise.keys())} noises"
            )

        if self.augmentation_reverb_args is not None:
            simulator_args = self.augmentation_reverb_args.get("simulator")
            if simulator_args and simulator_args.get("used"):
                pregenerated_args = simulator_args.get("pregenerated")
                if pregenerated_args and pregenerated_args.get("used"):
                    pregenerated_config = dict(pregenerated_args)
                    configured_role = pregenerated_config.get("usage_role")
                    if (
                        configured_role is not None
                        and str(configured_role) != self.dataset_role
                    ):
                        raise ValueError(
                            "pregenerated RIR usage_role does not match "
                            "the dataset_role"
                        )
                    pregenerated_config["usage_role"] = self.dataset_role
                    self.augmentor.init_room_bank(pregenerated_config)
                    mix = getattr(self.augmentor.room_bank, "describe", None)
                    print(
                        "Augmentor initialized pre-generated RIR bank "
                        f"(kind={self.augmentor.room_bank_kind}, "
                        f"items={len(self.augmentor.room_bank)}"
                        + (f", mix: {mix()}" if mix is not None else "")
                        + ")"
                    )
                else:
                    self.augmentor.init_room_simulator(simulator_args)
                    print("Augmentor initialized physics-based room simulator")
            else:
                self.augmentor.load_rir_from_folder(
                    self.augmentation_reverb_args["rir_folder"]
                )
                print(f"Augmentor finished load {len(self.augmentor.rir.keys())} rirs")

            drr_contrast_args = self.augmentation_reverb_args.get("drr_contrast")
            if drr_contrast_args and drr_contrast_args.get("used"):
                self.augmentor.init_drr_contrast(drr_contrast_args)
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
                print(
                    "Augmentor initialized DRR-contrast augmentation "
                    f"({detail}, window {knob['direct_window_ms']} ms)"
                )

        print("----" * 30)

    def should_apply_source_level_reverb(self) -> bool:
        if not (self.augmentation_reverb_args and self.augmentation_reverb_args["used"]):
            return False
        simulator_args = self.augmentation_reverb_args.get("simulator")
        if not (
            simulator_args
            and simulator_args.get("used")
            and simulator_args.get("source_level")
        ):
            return False
        return (torch.rand(1) < self.augmentation_reverb_args["prob"]).item()

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
        target_rir_type = self.augmentation_reverb_args["target_rir_type"]
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
                print(
                    f"Open an empty segment: {self.meta[target_speaker_name]['utts'][tgt_key]['path']}."
                )
                print(f"Retry {timeout} times.")
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
