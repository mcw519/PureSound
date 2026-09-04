"""P0 of the v20 programme: does the frozen bottleneck carry TALKER IDENTITY,
separably from the capture channel and from proximity?

The record before this probe: the foreground template is ~30 % channel signature
(`v19c_diagnostics/anchor_synthetic/TABLE_ident.md`: same-speaker-other-room prefix
-0.44 dB vs different-speaker -1.47 dB), proximity is readable on the device chain
(AUC 0.99) and inverted on QVF by the trained head while a device-fitted linear probe
reads QVF 0.79-0.83 (`v19c_diagnostics/chain_readability/`). Identity itself has never
been measured (`objective_landscape/LANDSCAPE.md`: "nobody has probed whether this
bottleneck separates talker identity from channel identity"). This script measures it.

Features are exactly what `anchor_gate_cache.py` caches: the TRAINING config's
backbone with `stash_bottleneck=True`, `last_bottleneck.mean(dim=2)` -> [128, T] at
100 fps. A segment embedding is the mean over SPEECH-ACTIVE frames of a 3 s
single-talker segment, activity taken from the close-talk (real) or dry (synthetic)
reference. Three feature variants: raw, per-frame LayerNorm over the 128 channels,
and per-recording mean-centred (computed at report time).

Nothing is trained on the features except the leave-speakers-out linear projection of
measurement B (LDA on the fit half's speaker labels, cosine in the projected space on
the held-out half).

    # index the segment lists (cheap, CPU)
    uv run python benchmarks/probes/identity_probe/identity_probe.py index --set dipco
    uv run python benchmarks/probes/identity_probe/identity_probe.py index --set ami
    uv run python benchmarks/probes/identity_probe/identity_probe.py index --set notsofar
    uv run python benchmarks/probes/identity_probe/identity_probe.py index --set synth

    # embeddings (one pass per model per set)
    uv run python benchmarks/probes/identity_probe/identity_probe.py extract \
        --set dipco --tag v16 --device cuda:1
    ... --tag v8 ... ; ... --tag sv ...        (sv = PS-spk-v1-1.onnx, CPU)

    # tables
    uv run python benchmarks/probes/identity_probe/identity_probe.py report
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import sys
import warnings
from pathlib import Path
from typing import Optional

#: This box is shared with other agents' training/benchmark jobs (24 cores). Torch's
#: default intra-op fan-out took ~6-12 cores per process and pushed the load average
#: past 30; the render/forward loop here is not helped by more than a few threads.
#: Set BEFORE numpy/torch import, and overridable from the environment.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, os.environ.get("P0_THREADS", "3"))

import numpy as np

warnings.filterwarnings("ignore")

REPO_ROOT = Path("/home/milowu/A4Audio/PureSound")
RECIPE_DIR = REPO_ROOT / "egs/voice_isolate"
SCRATCH = Path(
    "/tmp/claude-1001/-home-milowu-A4Audio-PureSound/"
    "c52f3b49-8e8f-4f9f-a5ca-862771bd494c/scratchpad/p0_identity"
)
OUT_DIR = Path(__file__).resolve().parent
TABLES = OUT_DIR / "tables"
CORPORA = RECIPE_DIR / "exp/real_e2e_corpora"
AMI_ANNOT = SCRATCH / "ami_annot"
LIBRITTS = Path("/data/audio/LibriTTS/test-clean")
BANK = Path("/work/any_exp_link/puresound_exp/hybrid_rir_16k_realfar")

SR = 16000
HOP = 160                 # bottleneck frame hop, 10 ms -> 100 fps
SEG_SEC = 3.0
SEG_LEN = int(SEG_SEC * SR)
ACTIVE_MARGIN_DB = 8.0    # over the reference's 5th-percentile floor
ACTIVE_REL_DB = 35.0      # ... and never more than this below its peak
MIN_ACTIVE_FRAMES = 100   # >= 1 s of speech in the 3 s segment
GAIN_DBFS = -28.0         # dataset `gain_normalized_to` of every voice_isolate recipe
SV_DBFS = -22.0           # `gain_normalized_to` of the speaker_embedding recipe
MAX_UTT_PER_SPEAKER = 10

CKPTS = {
    "v16": (
        "config/exp/train_dpcrn_v16_lengthmix.yaml",
        "/work/any_exp_link/puresound_exp/dpcrn_v16_lengthmix/lightning_logs/"
        "version_0/checkpoints/epoch=19-step=10000.ckpt",
    ),
    "v8": ("config/train_dpcrn.yaml", "pretrained_ckpt/dpcrn_v8.ckpt"),
}
SV_ONNX = REPO_ROOT / "egs/speaker_embedding/pretrained/PS-spk-v1-1.onnx"


# --------------------------------------------------------------------------- #
# activity / level helpers
# --------------------------------------------------------------------------- #
def frame_energy_db(x: np.ndarray, n_frames: int) -> np.ndarray:
    """20 ms frames on the 10 ms grid the bottleneck lives on (anchor_gate_sim)."""
    x = x.astype(np.float64)
    need = (n_frames - 1) * HOP + 2 * HOP
    if x.size < need:
        x = np.pad(x, (0, need - x.size))
    idx = np.arange(n_frames)[:, None] * HOP + np.arange(2 * HOP)[None, :]
    return 10.0 * np.log10((x[idx] ** 2).mean(axis=1) + 1e-12)


def activity_from_ref(ref: np.ndarray, n_frames: int) -> np.ndarray:
    e = frame_energy_db(ref, n_frames)
    thr = max(float(np.percentile(e, 5)) + ACTIVE_MARGIN_DB, float(e.max()) - ACTIVE_REL_DB)
    return e > thr


def rms_normalize(x: np.ndarray, dbfs: float) -> np.ndarray:
    r = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    if r < 1e-9:
        return x
    return (x * (10.0 ** (dbfs / 20.0) / r)).astype(np.float32)


def read_window(path: str, t0: float, n: int = SEG_LEN) -> np.ndarray:
    import soundfile as sf

    start = max(0, int(round(t0 * SR)))
    x, sr = sf.read(path, start=start, stop=start + n, dtype="float32", always_2d=False)
    if sr != SR:
        raise ValueError(f"{path}: {sr} != {SR}")
    if x.ndim > 1:
        x = x.mean(1)
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    return np.ascontiguousarray(x)


# --------------------------------------------------------------------------- #
# real-corpus segment indices
# --------------------------------------------------------------------------- #
def _merge(iv: list[tuple[float, float]], gap: float) -> list[tuple[float, float]]:
    if not iv:
        return []
    iv = sorted(iv)
    out = [list(iv[0])]
    for a, b in iv[1:]:
        if a - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def solo_windows(spans, min_len: float = SEG_SEC + 0.4, merge_gap: float = 0.6,
                 guard: float = 0.25, max_per_region: int = 2,
                 max_per_speaker: int = MAX_UTT_PER_SPEAKER):
    """spans: [(t0, t1, speaker)] from the corpus's own annotation.

    Merges each speaker's own turns across gaps <= `merge_gap`, subtracts every OTHER
    speaker's turns (dilated by `guard` on both sides), and cuts the surviving solo
    stretches into non-overlapping SEG_SEC windows. -> [(speaker, t0, region_id)]"""
    per: dict[str, list[tuple[float, float]]] = {}
    for a, b, s in spans:
        per.setdefault(s, []).append((a, b))
    out, counts = [], {}
    for s in sorted(per):
        others = _merge([(a - guard, b + guard) for a, b, o in spans if o != s], 0.0)
        for a, b in _merge(per[s], merge_gap):
            cuts = [(a, b)]
            for oa, ob in others:
                nxt = []
                for ca, cb in cuts:
                    if ob <= ca or oa >= cb:
                        nxt.append((ca, cb))
                        continue
                    if oa > ca:
                        nxt.append((ca, oa))
                    if ob < cb:
                        nxt.append((ob, cb))
                cuts = nxt
            for ri, (ca, cb) in enumerate(cuts):
                if cb - ca < min_len:
                    continue
                n = min(max_per_region, int((cb - ca - 0.2) // SEG_SEC))
                for k in range(n):
                    if counts.get(s, 0) >= max_per_speaker:
                        break
                    out.append((s, ca + 0.1 + k * SEG_SEC, f"{ca:.2f}_{ri}"))
                    counts[s] = counts.get(s, 0) + 1
    return out


def _hms(s: str) -> float:
    h, m, sec = s.split(":")
    return int(h) * 3600 + int(m) * 60 + float(sec)


def index_dipco() -> list[dict]:
    """DiPCo: 10 dinner-party sessions, 4 speakers each, speaker ids P01-P32 unique
    across the corpus. Channels = the speaker's own worn close-talk mic plus the five
    table arrays U01..U05 (CH1 of each). The per-device start/end fields in the release
    are numerically IDENTICAL to `close-talk` in every session (checked: max |offset| =
    0.000 s over all 10 sessions), i.e. one global clock for every channel."""
    root = CORPORA / "dipco/Dipco"
    out = []
    for tjson in sorted(glob.glob(str(root / "transcriptions/*/S*.json"))):
        utts = json.load(open(tjson))
        split = Path(tjson).parent.name
        sess = utts[0]["session_id"]
        spans = [(_hms(u["start_time"]["close-talk"]), _hms(u["end_time"]["close-talk"]),
                  u["speaker_id"]) for u in utts if "close-talk" in u.get("start_time", {})]
        for spk, t0, region in solo_windows(spans):
            ct = root / f"audio/{split}/{sess}_{spk}.wav"
            if not ct.exists():
                continue
            chans = [("CT", "close", str(ct))]
            for k in range(1, 6):
                far = root / f"audio/{split}/{sess}_U0{k}.CH1.wav"
                if far.exists():
                    chans.append((f"U0{k}", "far", str(far)))
            if len(chans) < 3:
                continue
            uid = f"dipco:{sess}:{spk}:{region}:{t0:.2f}"
            for name, kind, path in chans:
                out.append(dict(
                    sid=f"{uid}:{name}", corpus="dipco", session=sess,
                    speaker=f"dipco:{spk}", channel=f"dipco:{name}", chan_kind=kind,
                    utt=uid, path=path, t0=t0, ref_path=str(ct), ref_t0=t0, room=sess,
                ))
    return out


def index_ami(max_meetings: int = 60) -> list[dict]:
    """AMI: manual `segments/*.segments.xml` give per-agent speech spans and
    `corpusResources/meetings.xml` maps (meeting, agent) -> headset channel and the
    corpus-GLOBAL participant id, so speaker labels survive across meetings. Channels =
    the speaker's own headset and the single available far device Array1-01 (the
    download carries no second array). All channels are sample-synced."""
    mfile = AMI_ANNOT / "corpusResources/meetings.xml"
    if not mfile.is_file():
        raise SystemExit(f"missing {mfile}: extract corpusResources/ and segments/ "
                         f"from exp/real_e2e_corpora/ami/ami_public_manual_1.6.2.zip")
    text = mfile.read_text(encoding="latin-1")
    meetings: dict[str, list[tuple[str, int, str]]] = {}
    for block in re.findall(r"<meeting\b.*?</meeting>", text, flags=re.S):
        obs = re.search(r'observation="([^"]+)"', block)
        if not obs:
            continue
        spk = [(m.group(2), int(m.group(1)), m.group(3)) for m in re.finditer(
            r'<speaker[^>]*channel="(\d+)"[^>]*nxt_agent="([^"]+)"[^>]*global_name="([^"]+)"',
            block)]
        if spk:
            meetings[obs.group(1)] = spk
    root = CORPORA / "ami/amicorpus"
    out = []
    done = 0
    for meet in sorted(meetings):
        if done >= max_meetings:
            break
        adir = root / meet / "audio"
        arr = adir / f"{meet}.Array1-01.wav"
        if not arr.exists():
            continue
        agents = [(a, ch, g) for a, ch, g in meetings[meet]
                  if (adir / f"{meet}.Headset-{ch}.wav").exists()]
        if len(agents) < 2:
            continue
        spans, head_of = [], {}
        for agent, ch, gname in agents:
            sfile = AMI_ANNOT / f"segments/{meet}.{agent}.segments.xml"
            if not sfile.is_file():
                continue
            head_of[gname] = adir / f"{meet}.Headset-{ch}.wav"
            body = sfile.read_text(encoding="latin-1")
            for m in re.finditer(
                    r'transcriber_start="([\d.]+)"\s+transcriber_end="([\d.]+)"', body):
                spans.append((float(m.group(1)), float(m.group(2)), gname))
        if len(head_of) < 2:
            continue
        done += 1
        for gname, t0, region in solo_windows(spans):
            head = head_of.get(gname)
            if head is None:
                continue
            uid = f"ami:{meet}:{gname}:{region}:{t0:.2f}"
            for name, kind, path in (("HS", "close", str(head)),
                                     ("Array1-01", "far", str(arr))):
                out.append(dict(
                    sid=f"{uid}:{name}", corpus="ami", session=meet,
                    speaker=f"ami:{gname}", channel=f"ami:{name}", chan_kind=kind,
                    utt=uid, path=path, t0=t0, ref_path=str(head), ref_t0=t0, room=meet,
                ))
    return out


def index_notsofar(max_meetings: int = 20, max_devices: int = 3) -> list[dict]:
    """NOTSOFAR-1: conference-room meetings, several DISTINCT far devices per room
    (mc_plaza_0 / mc_rockfall_0..2) plus one close-talk mic per participant. Speaker
    labels are per-meeting aliases ("Peter"), so they are namespaced by meeting and
    never paired across meetings."""
    root = CORPORA / "notsofar/benchmark-datasets/train_set/240825.1_train/MTG"
    out = []
    for mdir in sorted(root.iterdir())[:max_meetings]:
        gt = mdir / "gt_transcription.json"
        dv = mdir / "devices.json"
        if not gt.exists() or not dv.exists():
            continue
        utts = json.load(open(gt))
        devs = [d for d in json.load(open(dv))
                if d.get("is_mc") and not d.get("is_close_talk")]
        far = []
        for d in devs[:max_devices]:
            p = mdir / str(d["wav_file_names"]).split(",")[0]
            if p.exists():
                far.append((p.parent.name, str(p)))
        if not far:
            continue
        spans = [(u["start_time"], u["end_time"], u["speaker_id"]) for u in utts]
        ct_of = {u["speaker_id"]: mdir / u["ct_wav_file_name"] for u in utts}
        for spk, t0, region in solo_windows(spans):
            ct = ct_of.get(spk)
            if ct is None or not ct.exists():
                continue
            uid = f"nsf:{mdir.name}:{spk}:{region}:{t0:.2f}"
            chans = [("CT", "close", str(ct))] + [(n, "far", p) for n, p in far]
            for name, kind, path in chans:
                out.append(dict(
                    sid=f"{uid}:{name}", corpus="notsofar", session=mdir.name,
                    speaker=f"nsf:{mdir.name}:{spk}", channel=f"nsf:{name}",
                    chan_kind=kind, utt=uid, path=path, t0=t0 + 0.2,
                    ref_path=str(ct), ref_t0=t0 + 0.2, room=mdir.name,
                ))
    return out


# --------------------------------------------------------------------------- #
# synthetic controlled renders
# --------------------------------------------------------------------------- #
#: Two capture chains, each a FIXED draw from the v16 recipe's own device-chain
#: distributions.  `seed` reseeds torch+random immediately before `DeviceChain.apply`,
#: so every utterance of a chain goes through numerically the same stages (the realised
#: parameters are written to tables/synth_chains.json by `index --set synth`).
SYNTH_CHAINS = {
    "A": dict(seed=101, src=None, hpf=100.0, floor_dbfs=-50.0),
    "B": dict(seed=202, src=8000, hpf=300.0, floor_dbfs=-42.0),
}
#: Three rooms from the v16 bank (`augmentation_reverb.simulator.pregenerated.folder`).
#: R1/R2 are simulated members with deliberately different RT60; R3 is a measured
#: (real-IR) member. `near`/`far` are the bank channel indices actually used.
SYNTH_ROOMS = {
    "R1": dict(item="wide_room_001516_000007", origin="sim", rt60=0.350,
               near=(1, 0.493), far=(2, 2.971)),
    "R2": dict(item="wide_room_000264_000017", origin="sim", rt60=0.700,
               near=(0, 0.540), far=(3, 3.443)),
    "R3": dict(item="real_ace_Office_1_0000", origin="real", rt60=0.332,
               near=(0, 0.987), far=(1, 2.680)),
}
#: chain x room x distance cells actually rendered: both chains at near in every room
#: (chain is the varied factor) and chain A at far in every room (distance is the
#: varied factor, chain held fixed).
SYNTH_CELLS = ([(c, r, "near") for c in SYNTH_CHAINS for r in SYNTH_ROOMS]
               + [("A", r, "far") for r in SYNTH_ROOMS])


def index_synth(min_utt: int = 6) -> list[dict]:
    spk_dirs = sorted(p for p in LIBRITTS.iterdir() if p.is_dir())
    out = []
    for sd in spk_dirs:
        wavs = sorted(str(p) for p in sd.rglob("*.wav"))
        keep = []
        import soundfile as sf
        for w in wavs:
            if sf.info(w).duration >= 4.0:
                keep.append(w)
            if len(keep) >= min_utt:
                break
        if len(keep) < min_utt:
            continue
        for k, w in enumerate(keep):
            uid = f"synth:{sd.name}:{k}"
            for chain, room, dist in SYNTH_CELLS:
                out.append(dict(
                    sid=f"{uid}:{chain}{room}{dist}", corpus="synth", session="synth",
                    speaker=f"synth:{sd.name}", channel=f"synth:{chain}{room}",
                    chan_kind={"near": "close", "far": "far"}[dist],
                    utt=uid, path=w, t0=-1.0, ref_path=w, ref_t0=-1.0,
                    room=room, chain=chain, dist=dist,
                ))
    return out


class SynthRenderer:
    """Renders one synthetic segment: LibriTTS utterance -> RMS -28 dBFS -> bank RIR
    (`wav_apply_rir` mode "full", which peak-normalises the impulse, so distance is
    carried by DRR/tail/tilt and NOT by level, exactly as the recipe intends) ->
    absolute capture floor -> the recipe's `DeviceChain`."""

    def __init__(self):
        import torch
        from puresound.audio.augmentation import AudioEffectAugmentor
        from puresound.audio.io import AudioIO
        from puresound.config.augmentation import (
            HighPassAugmentation, SimpleProbAugmentation, SourceRateAugmentation,
        )
        from puresound.task.device_chain import DeviceChain

        self.torch = torch
        self.AudioIO = AudioIO
        self.chains, self.realised = {}, {}
        for name, spec in SYNTH_CHAINS.items():
            kw = dict(
                ir_response=SimpleProbAugmentation(used=True, prob=1.0),
                hpf=HighPassAugmentation(used=True, prob=1.0, cutoff=[spec["hpf"]],
                                         prob_each=[1.0]),
            )
            if spec["src"] is not None:
                kw["src"] = SourceRateAugmentation(used=True, prob=1.0,
                                                   src_range=[spec["src"]], prob_each=[1.0])
            self.chains[name] = DeviceChain(AudioEffectAugmentor(), **kw)
        self.rirs: dict[str, "torch.Tensor"] = {}

    def _rir(self, room: str, dist: str):
        key = f"{room}/{dist}"
        if key not in self.rirs:
            spec = SYNTH_ROOMS[room]
            wav, sr = self.AudioIO.open(str(BANK / f"items/{spec['item']}.wav"))
            if sr != SR:
                raise ValueError(f"bank item {spec['item']}: {sr} != {SR}")
            ch = spec[dist][0]
            self.rirs[key] = wav[ch:ch + 1].clone()
        return self.rirs[key]

    def render(self, spec: dict, seg_seed: int):
        """-> (mixture [SEG_LEN], dry reference [SEG_LEN])"""
        from puresound.audio.impulse_response import wav_apply_rir

        torch = self.torch
        wav, sr = self.AudioIO.open(spec["path"], target_lvl=None, verbose=False)
        x = wav.mean(0).numpy() if wav.dim() > 1 else wav.view(-1).numpy()
        if sr != SR:
            import torchaudio
            x = torchaudio.functional.resample(
                torch.from_numpy(np.ascontiguousarray(x)).float(), sr, SR).numpy()
        # the SEG_LEN window with the most speech-active frames (LibriTTS pads silence)
        nf = max(1, len(x) // HOP - 1)
        act = activity_from_ref(x, nf)
        w = SEG_LEN // HOP
        if nf > w:
            cs = np.concatenate([[0], np.cumsum(act.astype(np.int64))])
            best = int(np.argmax(cs[w:] - cs[:-w]))
        else:
            best = 0
        x = x[best * HOP: best * HOP + SEG_LEN]
        if len(x) < SEG_LEN:
            x = np.pad(x, (0, SEG_LEN - len(x)))
        dry = rms_normalize(x, GAIN_DBFS)

        y = wav_apply_rir(torch.from_numpy(dry).view(1, -1), self._rir(spec["room"], spec["dist"]),
                          SR, rir_mode="full")
        cspec = SYNTH_CHAINS[spec["chain"]]
        rng = np.random.default_rng(seg_seed)
        floor = rng.standard_normal(SEG_LEN).astype(np.float32)
        floor *= 10.0 ** (cspec["floor_dbfs"] / 20.0) / float(np.sqrt(np.mean(floor ** 2)))
        y = y + torch.from_numpy(floor).view(1, -1)
        torch.manual_seed(cspec["seed"])
        random.seed(cspec["seed"])
        res = self.chains[spec["chain"]].apply(y, y.clone(), sample_rate=SR)
        self.realised[spec["chain"]] = {k: v for k, v in res.applied.items()
                                        if not (isinstance(v, float) and v != v)}
        out = res.noisy.view(-1).numpy()[:SEG_LEN]
        if len(out) < SEG_LEN:
            out = np.pad(out, (0, SEG_LEN - len(out)))
        return np.ascontiguousarray(out.astype(np.float32)), dry


# --------------------------------------------------------------------------- #
# embedding extraction
# --------------------------------------------------------------------------- #
def load_dpcrn(tag: str, device: str):
    import torch

    sys.path.insert(0, str(REPO_ROOT))
    from puresound.config import load_recipe
    from puresound.recipes import init_siso_model

    cfg, ckpt = CKPTS[tag]
    model = init_siso_model(load_recipe(str(RECIPE_DIR / cfg), expected_task="voice_isolation",
                                        expected_purpose="train").model)
    path = ckpt if os.path.isabs(ckpt) else str(RECIPE_DIR / ckpt)
    state = torch.load(path, map_location="cpu")
    model.reload_checkpoint(state.get("state_dict", state), load_loss_func=False)
    model = model.to(device).eval()
    model.backbone.stash_bottleneck = True
    return model


def pooled_bottleneck(model, x: np.ndarray, device: str) -> np.ndarray:
    """[128, T] -- exactly `anchor_gate_cache.run`'s `feat`."""
    import torch

    with torch.no_grad():
        model(torch.from_numpy(x).float().view(1, -1).to(device))
        return model.backbone.last_bottleneck.mean(dim=2)[0].float().cpu().numpy()


def _ln(feat: np.ndarray) -> np.ndarray:
    """LayerNorm over the 128 channels, per frame."""
    mu = feat.mean(axis=0, keepdims=True)
    sd = feat.std(axis=0, keepdims=True)
    return (feat - mu) / (sd + 1e-5)


def cmd_index(args) -> None:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    fn = {"dipco": index_dipco, "ami": index_ami, "notsofar": index_notsofar,
          "synth": index_synth}[args.set]
    segs = fn()
    path = SCRATCH / f"segments_{args.set}.jsonl"
    path.write_text("\n".join(json.dumps(s) for s in segs) + "\n")
    spk = sorted({s["speaker"] for s in segs})
    ch = sorted({s["channel"] for s in segs})
    print(f"{args.set}: {len(segs)} segments, {len(spk)} speakers, "
          f"{len({s['session'] for s in segs})} sessions, {len(ch)} channels")
    print(f"  channels: {ch}")
    print(f"  -> {path}")
    if args.set == "synth":
        r = SynthRenderer()
        for name in SYNTH_CHAINS:
            spec = next(s for s in segs if s.get("chain") == name)
            r.render(spec, 0)
        (TABLES / "synth_chains.json").write_text(json.dumps(
            {"chains": SYNTH_CHAINS, "rooms": SYNTH_ROOMS,
             "realised_device_chain": r.realised}, indent=1))
        print(f"  realised chain records -> {TABLES/'synth_chains.json'}")


def cmd_extract(args) -> None:
    import torch

    torch.set_num_threads(int(os.environ.get("P0_THREADS", "3")))
    segs = [json.loads(l) for l in open(SCRATCH / f"segments_{args.set}.jsonl")]
    if args.limit:
        segs = segs[: args.limit]
    renderer = SynthRenderer() if args.set == "synth" else None
    out: dict[str, list] = {k: [] for k in
                            ("emb", "emb_ln", "emb_rawlvl", "emb_ln_rawlvl", "n_active")}
    kept = []

    if args.tag == "sv":
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(os.environ.get("P0_THREADS", "3"))
        so.inter_op_num_threads = 1
        sess = ort.InferenceSession(str(SV_ONNX), sess_options=so,
                                    providers=["CPUExecutionProvider"])
        iname = sess.get_inputs()[0].name
    else:
        model = load_dpcrn(args.tag, args.device)

    for i, s in enumerate(segs):
        if renderer is not None:
            x, ref = renderer.render(s, seg_seed=1000 + i)
        else:
            x = read_window(s["path"], s["t0"])
            ref = read_window(s["ref_path"], s["ref_t0"])
        nf = SEG_LEN // HOP - 1
        act = activity_from_ref(ref, nf)
        if act.sum() < MIN_ACTIVE_FRAMES:
            continue
        if args.tag == "sv":
            e = sess.run(None, {iname: rms_normalize(x, SV_DBFS).reshape(1, -1)})[0][0]
            out["emb"].append(e.astype(np.float32))
            out["emb_ln"].append(e.astype(np.float32))
            e2 = sess.run(None, {iname: x.reshape(1, -1)})[0][0]
            out["emb_rawlvl"].append(e2.astype(np.float32))
            out["emb_ln_rawlvl"].append(e2.astype(np.float32))
        else:
            for suffix, xi in (("", rms_normalize(x, GAIN_DBFS)), ("_rawlvl", x)):
                feat = pooled_bottleneck(model, xi, args.device)
                m = act[: feat.shape[1]]
                if m.sum() < MIN_ACTIVE_FRAMES:
                    m = np.ones(feat.shape[1], dtype=bool)
                out[f"emb{suffix}"].append(feat[:, m].mean(axis=1).astype(np.float32))
                out[f"emb_ln{suffix}"].append(_ln(feat)[:, m].mean(axis=1).astype(np.float32))
        out["n_active"].append(int(act.sum()))
        kept.append(s)
        if (i + 1) % 400 == 0:
            print(f"  {i + 1}/{len(segs)}", flush=True)

    npz = SCRATCH / f"emb_{args.set}_{args.tag}.npz"
    np.savez(npz,
             meta=json.dumps(kept),
             **{k: np.asarray(v) for k, v in out.items()})
    print(f"{args.set}/{args.tag}: kept {len(kept)}/{len(segs)} segments -> {npz}")


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Tie-corrected Mann-Whitney AUC (same implementation as anchor_gate_sim.auc)."""
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, allv.size + 1)
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(cnt.size)
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    return (ranks[: pos.size].sum() - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size)


def eer(pos: np.ndarray, neg: np.ndarray) -> float:
    """O(n log n): FRR(t) = #pos < t, FAR(t) = #neg >= t, evaluated at every observed
    score. (The naive threshold loop is O(n^2) and makes the bootstrap unusable.)"""
    pos = np.sort(pos[np.isfinite(pos)])
    neg = np.sort(neg[np.isfinite(neg)])
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    u = np.unique(np.concatenate([pos, neg]))
    frr = np.searchsorted(pos, u, side="left") / pos.size
    far = (neg.size - np.searchsorted(neg, u, side="left")) / neg.size
    i = int(np.argmin(np.abs(far - frr)))
    return float((far[i] + frr[i]) / 2.0)


def cluster_bootstrap(pos, neg, pos_g, neg_g, b=1000, seed=0):
    """Percentile CIs for pair AUC and EER, resampling the CLUSTER (session for the real
    corpora, the anchor segment's speaker for the synthetic set) each pair belongs to.
    Both statistics come out of ONE resampling loop."""
    groups = sorted(set(pos_g) | set(neg_g))
    if len(groups) < 2:
        return (float("nan"),) * 4
    gi = {g: k for k, g in enumerate(groups)}
    pidx = [[] for _ in groups]
    nidx = [[] for _ in groups]
    for k, g in enumerate(pos_g):
        pidx[gi[g]].append(k)
    for k, g in enumerate(neg_g):
        nidx[gi[g]].append(k)
    pidx = [np.asarray(v, dtype=np.int64) for v in pidx]
    nidx = [np.asarray(v, dtype=np.int64) for v in nidx]
    rng = np.random.default_rng(seed)
    av, ev = [], []
    for _ in range(b):
        pick = rng.integers(0, len(groups), len(groups))
        pl = [pidx[j] for j in pick if pidx[j].size]
        nl = [nidx[j] for j in pick if nidx[j].size]
        if not pl or not nl:
            continue
        p, n = pos[np.concatenate(pl)], neg[np.concatenate(nl)]
        av.append(auc(p, n))
        ev.append(eer(p, n))
    av = np.asarray([v for v in av if np.isfinite(v)])
    ev = np.asarray([v for v in ev if np.isfinite(v)])
    if av.size < 20:
        return (float("nan"),) * 4
    return (float(np.percentile(av, 2.5)), float(np.percentile(av, 97.5)),
            float(np.percentile(ev, 2.5)), float(np.percentile(ev, 97.5)))


def median_ci(x, g, b=1000, seed=0):
    x = np.asarray(x, dtype=np.float64)
    groups = sorted(set(g))
    idx = {gg: np.flatnonzero(np.asarray(g) == gg) for gg in groups}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(b):
        pick = rng.integers(0, len(groups), len(groups))
        sel = np.concatenate([idx[groups[j]] for j in pick])
        vals.append(np.median(x[sel]))
    return float(np.median(x)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# --------------------------------------------------------------------------- #
# pair construction
# --------------------------------------------------------------------------- #
class Store:
    """Segment table + one embedding matrix per feature variant."""

    VARIANTS = ("raw", "ln", "centred", "raw_rawlvl")

    def __init__(self, npz: Path):
        d = np.load(npz, allow_pickle=True)
        self.meta = json.loads(str(d["meta"]))
        raw = d["emb"].astype(np.float64)
        self.E = {
            "raw": raw,
            "ln": d["emb_ln"].astype(np.float64),
            "raw_rawlvl": d["emb_rawlvl"].astype(np.float64),
        }
        # (iii) per-recording mean-centred. A "recording" = (session, channel [, dist]):
        # one capture condition, the unit whose offset the chain imposes. For the real
        # corpora a far channel is exactly one wav file; `CT` is a device CLASS (each
        # speaker wears their own unit) so its group pools that session's headsets --
        # noted in the README, and harmless because no arm pairs close with close.
        rec = [f"{m['session']}|{m['channel']}|{m.get('dist','')}" for m in self.meta]
        cen = raw.copy()
        for r in sorted(set(rec)):
            k = np.flatnonzero(np.asarray(rec) == r)
            cen[k] -= raw[k].mean(axis=0, keepdims=True)
        self.E["centred"] = cen
        self.rec = rec
        for k in self.E:
            n = np.linalg.norm(self.E[k], axis=1, keepdims=True)
            self.E[k] = self.E[k] / np.maximum(n, 1e-12)
        self.by_session: dict[str, list[int]] = {}
        for i, m in enumerate(self.meta):
            self.by_session.setdefault(m["session"], []).append(i)
        # integer code arrays, so pair construction is numpy and not a Python loop
        def codes(key, default=""):
            vals = [m.get(key, default) for m in self.meta]
            _, inv = np.unique(np.asarray(vals, dtype=object).astype(str),
                               return_inverse=True)
            return inv.astype(np.int32)
        self.A = {k: codes(src) for k, src in
                  (("spk", "speaker"), ("chan", "channel"), ("utt", "utt"),
                   ("chain", "chain"), ("room", "room"), ("dist", "dist"))}
        self.A["kind"] = codes("chan_kind")
        uniq = list(np.unique(np.asarray([m["chan_kind"] for m in self.meta])))
        self.kind_code = {k: uniq.index(k) for k in uniq}
        for k in ("close", "far"):
            self.kind_code.setdefault(k, -1)

    def cos(self, variant: str, a, b) -> np.ndarray:
        E = self.E[variant]
        return np.einsum("ij,ij->i", E[np.asarray(a)], E[np.asarray(b)])


EXHAUSTIVE_MAX = 600        # above this many segments in a session, sample pairs
_PAIR_CACHE: dict = {}


def make_pairs(store: Store, same_talker: bool, same_channel: bool,
               kinds: tuple[str, str], cap: int = 400, seed: int = 0,
               chain: Optional[str] = None, room: Optional[str] = None,
               dist: Optional[str] = None, cell_key=None):
    """Pairs inside one session (so the room, the device set and the recording epoch are
    fixed), always from DIFFERENT utterances (so no pair shares acoustic content), and
    with the two members' `chan_kind` matching `kinds` in either order. `chain` / `room`
    / `dist` optionally constrain those synthetic factors to "same" or "diff".

    Returns (i_idx, j_idx, cluster) -- cluster = the session by default, or whatever
    `cell_key` returns for the first member (the synthetic set has one session, so its
    bootstrap unit is the anchor speaker). Fully vectorised: the synthetic set is one
    session of ~2000 segments and a Python-level rejection loop over it is minutes."""
    ck = (id(store), same_talker, same_channel, tuple(kinds), chain, room, dist,
          cap, seed, id(cell_key))
    if ck in _PAIR_CACHE:
        return _PAIR_CACHE[ck]
    A = store.A
    k0, k1 = store.kind_code[kinds[0]], store.kind_code[kinds[1]]
    rel = {"same": True, "diff": False}

    def accept(ai, bi):
        m = A["utt"][ai] != A["utt"][bi]
        m &= (A["spk"][ai] == A["spk"][bi]) == same_talker
        m &= (A["chan"][ai] == A["chan"][bi]) == same_channel
        ka, kb = A["kind"][ai], A["kind"][bi]
        m &= ((ka == k0) & (kb == k1)) | ((ka == k1) & (kb == k0))
        for key, want in (("chain", chain), ("room", room), ("dist", dist)):
            if want is not None:
                m &= (A[key][ai] == A[key][bi]) == rel[want]
        return m

    I, J, G = [], [], []
    for sess, idx in store.by_session.items():
        idx = np.asarray(idx, dtype=np.int64)
        rng = np.random.default_rng(abs(hash((sess, seed))) % (2 ** 32))
        if idx.size <= EXHAUSTIVE_MAX:
            a, b = np.triu_indices(idx.size, k=1)
            ai, bi = idx[a], idx[b]
            keep = np.flatnonzero(accept(ai, bi))
            rng.shuffle(keep)
            keep = keep[:cap]
            pi, pj = ai[keep], bi[keep]
        else:
            pi, pj, seen = [], [], set()
            for _ in range(40):                     # 40 x 200k draws max
                if sum(len(x) for x in pi) >= cap:
                    break
                a = rng.integers(0, idx.size, 200_000)
                b = rng.integers(0, idx.size, 200_000)
                ok = (a != b) & accept(idx[a], idx[b])
                ka, kb = a[ok], b[ok]
                # de-duplicate unordered pairs across draws
                key = np.minimum(ka, kb).astype(np.int64) * idx.size + np.maximum(ka, kb)
                _, uq = np.unique(key, return_index=True)
                ka, kb, key = ka[uq], kb[uq], key[uq]
                fresh = np.array([k not in seen for k in key], dtype=bool) \
                    if key.size else np.zeros(0, dtype=bool)
                seen.update(key[fresh].tolist())
                pi.append(idx[ka[fresh]])
                pj.append(idx[kb[fresh]])
            pi = np.concatenate(pi)[:cap] if pi else np.zeros(0, dtype=np.int64)
            pj = np.concatenate(pj)[:cap] if len(pj) else np.zeros(0, dtype=np.int64)
        I.append(pi)
        J.append(pj)
        if cell_key is None:
            G.extend([sess] * len(pi))
        else:
            G.extend(cell_key(store.meta[k]) for k in pi)
    res = (np.concatenate(I) if I else np.zeros(0, dtype=np.int64),
           np.concatenate(J) if J else np.zeros(0, dtype=np.int64), G)
    _PAIR_CACHE[ck] = res
    return res


def arm(store: Store, variant: str, pos_spec: dict, neg_spec: dict, seed=0,
        cell_key=None, cap=400) -> dict:
    pi, pj, pg = make_pairs(store, seed=seed, cell_key=cell_key, cap=cap, **pos_spec)
    ni, nj, ng = make_pairs(store, seed=seed + 1, cell_key=cell_key, cap=cap, **neg_spec)
    if len(pi) == 0 or len(ni) == 0:
        return dict(n_pos=len(pi), n_neg=len(ni), auc=float("nan"), eer=float("nan"))
    ps = store.cos(variant, pi, pj)
    ns = store.cos(variant, ni, nj)
    lo, hi, elo, ehi = cluster_bootstrap(ps, ns, pg, ng, seed=seed)
    return dict(n_pos=int(len(pi)), n_neg=int(len(ni)), n_clusters=len(set(pg) | set(ng)),
                auc=auc(ps, ns), auc_lo=lo, auc_hi=hi,
                eer=eer(ps, ns), eer_lo=elo, eer_hi=ehi,
                pos_med=float(np.median(ps)), neg_med=float(np.median(ns)))


# --------------------------------------------------------------------------- #
# measurement B: leave-speakers-out linear projection
# --------------------------------------------------------------------------- #
def linear_probe(store: Store, variant: str, pos_spec: dict, neg_spec: dict,
                 split_by: str, n_splits: int = 20, n_comp: int = 48, seed: int = 0,
                 cap: int = 400):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    units = sorted({store.meta[i][split_by] for i in range(len(store.meta))})
    if len(units) < 4:
        return dict(auc=float("nan"), eer=float("nan"), n_splits=0)
    pi, pj, _ = make_pairs(store, seed=seed, cap=cap, **pos_spec)
    ni, nj, _ = make_pairs(store, seed=seed + 1, cap=cap, **neg_spec)
    aucs, eers, nfit, ntest = [], [], [], []
    rng = random.Random(seed)
    for r in range(n_splits):
        u = list(units)
        rng.shuffle(u)
        half = len(u) // 2
        fit_u, test_u = set(u[:half]), set(u[half:])
        fit_i = [i for i in range(len(store.meta)) if store.meta[i][split_by] in fit_u]
        fit_spk = {store.meta[i]["speaker"] for i in fit_i}
        # speaker-disjointness even when the split unit is the session (AMI's global
        # participants recur across meetings): a speaker seen in the fit half is not
        # allowed to appear in the tested pairs.
        def testable(k):
            a, b = store.meta[k[0]], store.meta[k[1]]
            return (a[split_by] in test_u and b[split_by] in test_u
                    and a["speaker"] not in fit_spk and b["speaker"] not in fit_spk)

        pk = [k for k in zip(pi, pj) if testable(k)]
        nk = [k for k in zip(ni, nj) if testable(k)]
        y = [store.meta[i]["speaker"] for i in fit_i]
        if len(set(y)) < 3 or len(pk) < 20 or len(nk) < 20:
            continue
        X = store.E[variant][fit_i]
        lda = LinearDiscriminantAnalysis(
            solver="eigen", shrinkage="auto",
            n_components=min(n_comp, len(set(y)) - 1))
        lda.fit(X, y)
        Z = lda.transform(store.E[variant])
        Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
        ps = np.einsum("ij,ij->i", Z[[k[0] for k in pk]], Z[[k[1] for k in pk]])
        ns = np.einsum("ij,ij->i", Z[[k[0] for k in nk]], Z[[k[1] for k in nk]])
        aucs.append(auc(ps, ns))
        eers.append(eer(ps, ns))
        nfit.append(len(set(y)))
        ntest.append(len({store.meta[k[0]]["speaker"] for k in pk}
                         | {store.meta[k[1]]["speaker"] for k in nk}))
    if not aucs:
        return dict(auc=float("nan"), eer=float("nan"), n_splits=0)
    a = np.asarray(aucs)
    e = np.asarray(eers)
    return dict(auc=float(a.mean()), auc_lo=float(np.percentile(a, 2.5)),
                auc_hi=float(np.percentile(a, 97.5)), auc_sd=float(a.std()),
                eer=float(e.mean()), eer_lo=float(np.percentile(e, 2.5)),
                eer_hi=float(np.percentile(e, 97.5)), n_splits=len(aucs),
                n_fit_spk=int(np.median(nfit)), n_test_spk=int(np.median(ntest)))


# --------------------------------------------------------------------------- #
# arms
# --------------------------------------------------------------------------- #
def far_far(kinds=("far", "far")):
    return kinds


#: A: primary identity arm. positives = same talker through DIFFERENT devices,
#: negatives = different talkers through the SAME device.  Two proximity regimes:
#:   `matched` -- both members far (two table arrays / two synthetic near chains):
#:                proximity class is held constant, so only the device differs.
#:   `xprox`   -- close-talk vs far device: the corpora's own pairing, which changes
#:                chain AND proximity together.
ARMS = {
    "A_matched": dict(
        pos=dict(same_talker=True, same_channel=False, kinds=("far", "far")),
        neg=dict(same_talker=False, same_channel=True, kinds=("far", "far"))),
    "A_xprox": dict(
        pos=dict(same_talker=True, same_channel=False, kinds=("close", "far")),
        neg=dict(same_talker=False, same_channel=True, kinds=("far", "far"))),
    # symmetric protocol: BOTH trial types cross the channel, so the channel change is
    # a nuisance on both sides instead of only on the positives
    "A_xchan_both": dict(
        pos=dict(same_talker=True, same_channel=False, kinds=("far", "far")),
        neg=dict(same_talker=False, same_channel=False, kinds=("far", "far"))),
    "A_xprox_both": dict(
        pos=dict(same_talker=True, same_channel=False, kinds=("close", "far")),
        neg=dict(same_talker=False, same_channel=False, kinds=("close", "far"))),
    "C1_chan_fixed": dict(
        pos=dict(same_talker=True, same_channel=True, kinds=("far", "far")),
        neg=dict(same_talker=False, same_channel=True, kinds=("far", "far"))),
    "C2_channel_move": dict(
        pos=dict(same_talker=True, same_channel=True, kinds=("far", "far")),
        neg=dict(same_talker=True, same_channel=False, kinds=("far", "far"))),
    "C3_channel_vs_talker": dict(
        pos=dict(same_talker=False, same_channel=True, kinds=("far", "far")),
        neg=dict(same_talker=True, same_channel=False, kinds=("far", "far"))),
}

CELLS_2x2 = {
    "sameT_sameC": dict(same_talker=True, same_channel=True, kinds=("far", "far")),
    "sameT_diffC": dict(same_talker=True, same_channel=False, kinds=("far", "far")),
    "diffT_sameC": dict(same_talker=False, same_channel=True, kinds=("far", "far")),
    "diffT_diffC": dict(same_talker=False, same_channel=False, kinds=("far", "far")),
    "sameT_xprox": dict(same_talker=True, same_channel=False, kinds=("close", "far")),
    "diffT_xprox": dict(same_talker=False, same_channel=False, kinds=("close", "far")),
}

#: D (synthetic only). `chain` and `room` are separate factors, so the two ways of being
#: a different device can be told apart, and distance is varied with the chain held
#: fixed. Near and far are the SAME capture setup, so a near/far pair has
#: `same_channel=True` and differs only in `dist`.
_NN = ("close", "close")
_NF = ("close", "far")

SYNTH_ARMS = {
    # identity across a chain swap, room and distance held fixed
    "D1_chain_swap": dict(
        pos=dict(same_talker=True, same_channel=False, kinds=_NN,
                 chain="diff", room="same", dist="same"),
        neg=dict(same_talker=False, same_channel=True, kinds=_NN,
                 chain="same", room="same", dist="same")),
    # identity across a room swap, chain and distance held fixed
    "D2_room_swap": dict(
        pos=dict(same_talker=True, same_channel=False, kinds=_NN,
                 chain="same", room="diff", dist="same"),
        neg=dict(same_talker=False, same_channel=True, kinds=_NN,
                 chain="same", room="same", dist="same")),
    # identity across chain AND room (the real corpora's "different channel")
    "D3_chain_and_room": dict(
        pos=dict(same_talker=True, same_channel=False, kinds=_NN,
                 chain="diff", room="diff", dist="same"),
        neg=dict(same_talker=False, same_channel=True, kinds=_NN,
                 chain="same", room="same", dist="same")),
    # identity across a DISTANCE change, chain and room held fixed
    "D4_distance": dict(
        pos=dict(same_talker=True, same_channel=True, kinds=_NF,
                 chain="same", room="same", dist="diff"),
        neg=dict(same_talker=False, same_channel=True, kinds=_NN,
                 chain="same", room="same", dist="same")),
    # symmetric protocol: both trial types cross the chain
    "D5_xchain_both": dict(
        pos=dict(same_talker=True, same_channel=False, kinds=_NN,
                 chain="diff", room="same", dist="same"),
        neg=dict(same_talker=False, same_channel=False, kinds=_NN,
                 chain="diff", room="same", dist="same")),
    # control: identity with chain, room and distance ALL fixed
    "D0_all_fixed": dict(
        pos=dict(same_talker=True, same_channel=True, kinds=_NN,
                 chain="same", room="same", dist="same"),
        neg=dict(same_talker=False, same_channel=True, kinds=_NN,
                 chain="same", room="same", dist="same")),
}

SYNTH_CELLS_2x2 = {
    "sameT_sameChain_sameRoom": dict(same_talker=True, same_channel=True, kinds=_NN,
                                     chain="same", room="same", dist="same"),
    "sameT_diffChain_sameRoom": dict(same_talker=True, same_channel=False, kinds=_NN,
                                     chain="diff", room="same", dist="same"),
    "sameT_sameChain_diffRoom": dict(same_talker=True, same_channel=False, kinds=_NN,
                                     chain="same", room="diff", dist="same"),
    "sameT_near_far": dict(same_talker=True, same_channel=True, kinds=_NF,
                           chain="same", room="same", dist="diff"),
    "diffT_sameChain_sameRoom": dict(same_talker=False, same_channel=True, kinds=_NN,
                                     chain="same", room="same", dist="same"),
    "diffT_diffChain_sameRoom": dict(same_talker=False, same_channel=False, kinds=_NN,
                                     chain="diff", room="same", dist="same"),
    "diffT_sameChain_diffRoom": dict(same_talker=False, same_channel=False, kinds=_NN,
                                     chain="same", room="diff", dist="same"),
    "diffT_near_far": dict(same_talker=False, same_channel=True, kinds=_NF,
                           chain="same", room="same", dist="diff"),
}


def cmd_report(args) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    tags = args.tags.split(",")
    sets = args.sets.split(",")
    rows = []
    for st in sets:
        for tag in tags:
            npz = SCRATCH / f"emb_{st}_{tag}.npz"
            if not npz.is_file():
                print(f"skip {st}/{tag}: no {npz.name}")
                continue
            store = Store(npz)
            _PAIR_CACHE.clear()      # keys hold id(store); never reuse across stores
            synth = st == "synth"
            cluster = (lambda m: m["speaker"]) if synth else None
            arms = SYNTH_ARMS if synth else ARMS
            cells = SYNTH_CELLS_2x2 if synth else CELLS_2x2
            cap = 6000 if synth else 400
            variants = ("raw", "ln", "centred", "raw_rawlvl")
            for name, spec in arms.items():
                for v in variants:
                    r = arm(store, v, spec["pos"], spec["neg"], cell_key=cluster, cap=cap)
                    rows.append(dict(table="A" if name.startswith(("A", "D")) else "C",
                                     set=st, tag=tag, arm=name, variant=v, **r))
            for cname, spec in cells.items():
                i, j, g = make_pairs(store, cell_key=cluster, cap=cap, **spec)
                for v in variants:
                    if len(i) == 0:
                        continue
                    s = store.cos(v, i, j)
                    med, lo, hi = median_ci(s, g)
                    rows.append(dict(table="2x2", set=st, tag=tag, arm=cname, variant=v,
                                     n_pos=int(len(i)), med=med, med_lo=lo, med_hi=hi))
            # B: leave-speakers-out linear projection, on the same pair definition as
            # the set's primary A/D arm
            for key in (["D3_chain_and_room", "D1_chain_swap"] if synth
                        else ["A_matched", "A_xprox"]):
                if key not in arms:
                    continue
                # "speaker" = strict leave-speakers-out; "session" additionally holds
                # out the whole recording (no channel instance shared with the fit half)
                for split_by in (["speaker"] if synth else ["speaker", "session"]):
                    for v in ("raw", "ln", "centred"):
                        r = linear_probe(store, v, arms[key]["pos"], arms[key]["neg"],
                                         split_by=split_by, cap=cap)
                        rows.append(dict(table="B", set=st, tag=tag, arm=key, variant=v,
                                         split_by=split_by, **r))
            print(f"done {st}/{tag} ({len(store.meta)} segments)")
    (TABLES / "results.json").write_text(json.dumps(rows, indent=1))
    write_markdown(rows, tags, sets)
    print(f"-> {TABLES/'results.json'} ({len(rows)} rows)")
    print(f"\n{'set':9s} {'model':5s} {'arm':22s} {'var':11s} {'AUC':>6s} "
          f"{'CI':>16s} {'EER':>6s} {'npos':>6s} {'nneg':>6s}")
    for r in rows:
        if r["table"] not in ("A", "C") or r["variant"] == "raw_rawlvl":
            continue
        print(f"{r['set']:9s} {r['tag']:5s} {r['arm']:22s} {r['variant']:11s} "
              f"{_f(r.get('auc')):>6s} [{_f(r.get('auc_lo'))},{_f(r.get('auc_hi'))}]"
              f" {_f(r.get('eer')):>6s} {r.get('n_pos'):>6} {r.get('n_neg'):>6}")
    print(f"\n{'set':9s} {'model':5s} {'B arm':22s} {'holdout':8s} {'var':9s} "
          f"{'AUC':>6s} {'spl':>4s}")
    for r in rows:
        if r["table"] != "B":
            continue
        print(f"{r['set']:9s} {r['tag']:5s} {r['arm']:22s} {str(r.get('split_by')):8s} "
              f"{r['variant']:9s} {_f(r.get('auc')):>6s} {r.get('n_splits'):>4}")


# --------------------------------------------------------------------------- #
# markdown rendering
# --------------------------------------------------------------------------- #
def _f(v, nd=3):
    if v is None or (isinstance(v, float) and v != v):
        return "--"
    return f"{v:.{nd}f}"


def _ci(r, key):
    lo, hi = r.get(f"{key}_lo"), r.get(f"{key}_hi")
    if lo is None or hi is None or lo != lo or hi != hi:
        return _f(r.get(key))
    return f"{_f(r.get(key))} [{_f(lo)}, {_f(hi)}]"


ARM_DESC = {
    "A_matched": "pos same talker / 2 different far devices; neg different talkers / same far device (proximity matched)",
    "A_xprox": "pos same talker / close-talk vs far device; neg different talkers / same far device",
    "A_xchan_both": "pos same talker, neg different talkers -- BOTH across 2 different far devices (symmetric SV protocol)",
    "A_xprox_both": "pos same talker, neg different talkers -- BOTH close-talk vs far device",
    "D5_xchain_both": "pos same talker, neg different talkers -- BOTH across the 2 device chains, room+distance fixed",
    "C1_chan_fixed": "pos same talker / SAME far device; neg different talkers / same far device (channel + proximity fixed)",
    "C2_channel_move": "pos same talker same device; neg SAME talker different device (how far the channel moves a talker)",
    "C3_channel_vs_talker": "pos different talkers SAME device; neg same talker different device (channel vs identity)",
    "D0_all_fixed": "pos same talker, chain+room+distance fixed; neg different talkers, same cell",
    "D1_chain_swap": "pos same talker across the 2 device chains, room+distance fixed; neg different talkers same cell",
    "D2_room_swap": "pos same talker across 2 rooms, chain+distance fixed; neg different talkers same cell",
    "D3_chain_and_room": "pos same talker, chain AND room both change; neg different talkers same cell",
    "D4_distance": "pos same talker near vs far, chain+room fixed; neg different talkers, both near",
}
VAR_DESC = {"raw": "(i) raw pooled", "ln": "(ii) per-frame LayerNorm over channels",
            "centred": "(iii) per-recording mean-centred",
            "raw_rawlvl": "(i) raw, file-native level (no RMS normalisation)"}


def write_markdown(rows, tags, sets) -> None:
    L = ["# P0 identity probe -- raw tables",
         "",
         "Generated by `identity_probe.py report`. `AUC` is the tie-corrected pair AUC "
         "(P(same-talker pair scores higher)); CIs are 95 % percentile intervals from a "
         "1000-draw cluster bootstrap over sessions (real corpora) or over the anchor "
         "speaker (synthetic set). Feature variants: "
         + "; ".join(f"**{k}** {v}" for k, v in VAR_DESC.items()) + ".",
         ""]
    for table, title in (("A", "Tables A / D -- zero-shot cosine pair test"),
                         ("C", "Table C -- confound controls"),
                         ("B", "Table B -- leave-speakers-out linear projection (LDA)")):
        L += [f"## {title}", ""]
        if table == "B":
            L += ["| set | model | arm | held out by | variant | splits | fit spk | "
                  "test spk | AUC (mean) [2.5, 97.5 over splits] | EER (mean) |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
            for r in rows:
                if r["table"] != "B":
                    continue
                L.append(f"| {r['set']} | {r['tag']} | {r['arm']} | "
                         f"{r.get('split_by')} | {r['variant']} | "
                         f"{r.get('n_splits')} | {r.get('n_fit_spk','--')} | "
                         f"{r.get('n_test_spk','--')} | {_ci(r,'auc')} | {_f(r.get('eer'))} |")
        else:
            L += ["| set | model | arm | variant | n pos | n neg | clusters | "
                  "pair AUC [95% CI] | EER [95% CI] | med cos pos | med cos neg |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
            for r in rows:
                if r["table"] != table:
                    continue
                L.append(f"| {r['set']} | {r['tag']} | {r['arm']} | {r['variant']} | "
                         f"{r.get('n_pos')} | {r.get('n_neg')} | {r.get('n_clusters')} | "
                         f"{_ci(r,'auc')} | {_ci(r,'eer')} | {_f(r.get('pos_med'))} | "
                         f"{_f(r.get('neg_med'))} |")
        L.append("")
    L += ["### arm definitions", ""]
    for k, v in ARM_DESC.items():
        L.append(f"* `{k}` -- {v}")
    L += ["", "## 2x2 median cosine (same/different talker x same/different channel)", ""]
    cells = [r["arm"] for r in rows if r["table"] == "2x2"]
    order = list(dict.fromkeys(cells))
    for st in sets:
        for v in ("raw", "ln", "centred", "raw_rawlvl"):
            sub = [r for r in rows if r["table"] == "2x2" and r["set"] == st
                   and r["variant"] == v]
            if not sub:
                continue
            models = [t for t in tags if any(r["tag"] == t for r in sub)]
            L += [f"**{st} / {v}**", "",
                  "| cell | n pairs | " + " | ".join(f"{m} median cos [95% CI]"
                                                     for m in models) + " |",
                  "|---|---|" + "---|" * len(models)]
            for c in order:
                rr = {r["tag"]: r for r in sub if r["arm"] == c}
                if not rr:
                    continue
                n = next(iter(rr.values())).get("n_pos")
                L.append(f"| {c} | {n} | "
                         + " | ".join(_ci(rr[m], "med") if m in rr else "--"
                                      for m in models) + " |")
            L.append("")
    (TABLES / "TABLES.md").write_text("\n".join(L) + "\n")
    print(f"-> {TABLES/'TABLES.md'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("index")
    p.add_argument("--set", required=True,
                   choices=["dipco", "ami", "notsofar", "synth"])
    p.set_defaults(fn=cmd_index)
    p = sub.add_parser("extract")
    p.add_argument("--set", required=True)
    p.add_argument("--tag", required=True, choices=["v8", "v16", "sv"])
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--limit", type=int, default=None)
    p.set_defaults(fn=cmd_extract)
    p = sub.add_parser("report")
    p.add_argument("--tags", default="v16,v8,sv")
    p.add_argument("--sets", default="dipco,ami,notsofar,synth")
    p.set_defaults(fn=cmd_report)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
