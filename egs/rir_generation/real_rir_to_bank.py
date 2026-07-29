"""Convert a REAL measured RIR dataset into a PreGeneratedRoomBank folder.

Simulated RIRs leave a gap between in-domain separation and real-recording
performance. Measured RIRs with a known source-receiver DISTANCE let a bank be
built whose near/far structure matches the training task while its acoustics come
from real rooms, which closes the RIR part of that gap.

Two stages, decoupled by a manifest so the (definitive, self-tested) bank-format
emission is independent of any one dataset's on-disk layout:

  Stage A  dataset -> manifest.jsonl   (thin, dataset-specific scanner; ``scan``
                                        covers the public corpora listed below,
                                        ``but`` runs Stage A+B for BUT ReverbDB)
  Stage B  manifest.jsonl -> bank/     (THIS is the part that must be exactly
                                        right; it is self-tested against
                                        PreGeneratedRoomBank)

Stage-A corpora covered by ``scan`` (each has its own geometry quirk, noted because
getting the distance wrong silently poisons the near/far labels):

  dechorate  SOFA re-release (sofacoustics.org) -- the Zenodo hdf5 is corrupted
             (github.com/Chutlhu/dEchorate #35/#37/#39). Calibrated coordinates
             per file; 11 panel configs of one 6x6x2.4 m lab, treated as 11
             acoustic rooms. Distance from SourcePosition-ReceiverPosition.
  brudex     v7.3 MAT RIRs; channels 0-5 head-mounted (skipped: unknown HRTF
             coloring), 6-17 eMics E{(run-1)*12+1..run*12}. Distance is taken
             from the direct-path TOA (verified: absolute timing is preserved,
             onset/(fs/c) matches the nominal Setup_Database.pdf geometry to
             ~0.1-0.2 m for the vast majority of channels). The NOMINAL
             geometry (eMic grid x=0.75+0.95i, y=0.80+0.95j from bottom-left
             E1 row-major, KEMAR head (2.90, 3.49), sources on a 2.0 m circle,
             theta from +y, +90 deg = +x, all z=1.5 m) serves as cross-check:
             channels where |TOA - nominal| > 0.5 m are DROPPED (dead channel
             / blocked direct path / mapping mismatch, a few percent).
  slr28      openSLR-28 REVERB real RIRs (RVB2014_type1_rir_*): 6 real rooms
             x near(0.5 m)/far(2.0 m) x 2 angles x 8 ch. Distances documented.
             (RWCP fixed-2m and AIR binaural files are skipped.)
  ace        ACE Single-mic pack: 7 rooms x 2 positions, 1 file each.
             Distances from the ACE tech report (arXiv 1606.03365) Table 4,
             column "Ch-1 of 8-ch. lin." -- ACE itself defines the Single
             config as channel 1 of the linear array. (RIR onset TOA proved
             unusable: per-session equipment latency is not constant.)
  diffrir    Hearing-Anything-Anywhere zips: per config, RIRs from npy + mic
             xyzs.npy; speaker_xyz per config from the paper repo (values
             hardcoded below, incl. panel/rotation/translation variants).

Manifest line schema (one JSON object per line):
    {
      "room_id":   "Room_A",            # items are assembled within a room_id
      "rir_path":  "/abs/path.wav",     # mono or multi-channel RIR file
      "channel":   0,                   # channel index in rir_path (default 0)
      "rt60":      0.45,                # room RT60 (s); optional
      # distance is taken from "distance_m" if present, else ||src-mic||:
      "distance_m": 1.2,                # optional
      "src_xyz":   [x, y, z],           # optional (loudspeaker, by reciprocity)
      "mic_xyz":   [x, y, z]            # optional (microphone)
    }

Reciprocity: a measured RIR is identical whether read as source->mic or
mic->source, so a dataset with ONE loudspeaker and MANY microphones (BUT, UPV,
...) maps cleanly onto the bank's "one receiver, many sources at various
distances" item -- the distance is just ||loudspeaker - microphone||.

Bank item emitted per assembled scene:
    <out>/<item_id>/<item_id>.wav   # [C, T] float, one RIR per channel
    <out>/<item_id>/<item_id>.json  # {"scene": {"rt60", "channel_map":[...]}}
where channel_map[i] = {"channel": i, "label": "near_0"|.., "distance_m": d}.

Usage:
    # Stage B only (you supply the manifest):
    python real_rir_to_bank.py from-manifest \
        --manifest manifest.jsonl --output exp/real_rir_bank \
        --d0 1.0 --target-sr 16000

    # BUT ReverbDB end-to-end (scan -> manifest -> bank). The scanner is
    # best-effort; verify it against the downloaded layout (see scan_but()).
    python real_rir_to_bank.py but \
        --input /path/to/BUT_ReverbDB --output exp/real_rir_bank --d0 1.0

    # Self-test the definitive Stage B (no dataset needed):
    python real_rir_to_bank.py self-test
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
import torchaudio


# ----------------------------------------------------------------------------
# Stage B -- manifest -> bank  (definitive, self-tested)
# ----------------------------------------------------------------------------

NEAR_LABELS = ("near_0", "near_1")
FAR_LABELS = ("far_0", "far_1", "far_2")


def _entry_distance(entry: dict) -> float:
    if entry.get("distance_m") is not None:
        return float(entry["distance_m"])
    src, mic = entry.get("src_xyz"), entry.get("mic_xyz")
    if src is None or mic is None:
        raise ValueError(
            f"manifest entry needs 'distance_m' or both 'src_xyz' and 'mic_xyz': {entry}"
        )
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(src, mic)))


def _load_rir_mono(rir_path: str, channel: int, target_sr: int) -> torch.Tensor:
    wav, sr = torchaudio.load(rir_path)  # [C, T]
    ch = min(max(int(channel), 0), wav.shape[0] - 1)
    wav = wav[ch : ch + 1]  # [1, T]
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav  # [1, T]


def assemble_room_items(
    room_entries: list[dict],
    d0: float,
    items_per_room: int,
    near_max: int = 2,
    far_max: int = 3,
    require_near: int = 1,
    require_far: int = 1,
    rng: torch.Generator | None = None,
    near_pool: list[dict] | None = None,
) -> list[list[tuple[dict, str]]]:
    """Split a room's RIR entries into near/far by distance and assemble items.

    Each returned item is a list of ``(entry, label)`` with labels near_0.. and
    far_0.. . Rooms without enough near/far entries (per require_*) yield
    nothing -- unless ``near_pool`` is given: rooms with far entries but NO
    near entries then borrow near channels from the pool (real <d0 RIRs from
    other rooms/corpora). Physically the borrowed near channel belongs to a
    different room, but a <1 m channel is direct-path dominated so the
    room-mismatch contribution is small -- this is what lets far-only measured
    corpora (ACE, DIFFRIR classroom/complex) contribute their real far field.
    """
    rng = rng or torch.Generator().manual_seed(0)
    near = sorted((e for e in room_entries if _entry_distance(e) < d0),
                  key=_entry_distance)
    far = sorted((e for e in room_entries if _entry_distance(e) >= d0),
                 key=_entry_distance)
    if not near and far and near_pool:
        near = list(near_pool)
    if len(near) < require_near or len(far) < require_far:
        return []

    def _sample(pool: list[dict], k: int) -> list[dict]:
        k = min(k, len(pool))
        idx = torch.randperm(len(pool), generator=rng)[:k].tolist()
        return [pool[i] for i in idx]

    items: list[list[tuple[dict, str]]] = []
    for _ in range(items_per_room):
        n_near = int(torch.randint(require_near, near_max + 1, (1,), generator=rng))
        n_far = int(torch.randint(require_far, far_max + 1, (1,), generator=rng))
        chosen = [(e, NEAR_LABELS[i]) for i, e in enumerate(_sample(near, n_near))]
        chosen += [(e, FAR_LABELS[i]) for i, e in enumerate(_sample(far, n_far))]
        items.append(chosen)
    return items


def write_bank_item(
    out_dir: Path,
    item_id: str,
    channels: list[tuple[dict, str]],
    target_sr: int,
    rt60: float | None,
) -> None:
    """Write one bank item: multi-channel WAV + channel_map JSON."""
    item_dir = out_dir / item_id
    item_dir.mkdir(parents=True, exist_ok=True)

    rirs = [_load_rir_mono(e["rir_path"], e.get("channel", 0), target_sr)
            for e, _label in channels]
    length = min(r.shape[-1] for r in rirs)
    wav = torch.cat([r[..., :length] for r in rirs], dim=0)  # [C, T]

    channel_map = [
        {"channel": i, "label": label, "distance_m": _entry_distance(e)}
        for i, (e, label) in enumerate(channels)
    ]
    # origin: "real" marks measured-RIR items so the training pipeline can gate
    # per-origin augmentation probs (ns.py prob_by_origin / turn_taking_prob_by_origin).
    scene = {"channel_map": channel_map, "origin": "real"}
    if rt60 is not None:
        scene["rt60"] = float(rt60)

    torchaudio.save(str(item_dir / f"{item_id}.wav"), wav, target_sr,
                    encoding="PCM_F", bits_per_sample=32)
    (item_dir / f"{item_id}.json").write_text(
        json.dumps({"scene": scene}, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _read_manifest(manifest_path: str) -> list[dict]:
    entries: list[dict] = []
    with open(manifest_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def manifest_to_bank(
    manifest_path: str,
    output: str,
    d0: float,
    target_sr: int,
    items_per_room: int,
    seed: int = 0,
    near_pool_manifest: str | None = None,
) -> dict:
    rooms: dict[str, list[dict]] = {}
    for e in _read_manifest(manifest_path):
        rooms.setdefault(str(e["room_id"]), []).append(e)

    near_pool: list[dict] | None = None
    if near_pool_manifest is not None:
        near_pool = [e for e in _read_manifest(near_pool_manifest)
                     if _entry_distance(e) < d0]
        if not near_pool:
            raise ValueError(f"near-pool manifest has no entries < d0={d0}")

    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = torch.Generator().manual_seed(seed)
    n_items = n_rooms_used = 0
    for room_id, entries in sorted(rooms.items()):
        rt60s = [e["rt60"] for e in entries if e.get("rt60") is not None]
        rt60 = (sum(rt60s) / len(rt60s)) if rt60s else None
        items = assemble_room_items(entries, d0, items_per_room, rng=rng,
                                    near_pool=near_pool)
        if items:
            n_rooms_used += 1
        for k, channels in enumerate(items):
            write_bank_item(out_dir, f"{room_id}_{k:04d}", channels, target_sr, rt60)
            n_items += 1
    summary = {
        "rooms_total": len(rooms),
        "rooms_used": n_rooms_used,
        "items_written": n_items,
        "d0": d0,
        "target_sr": target_sr,
        "output": str(out_dir),
    }
    print(json.dumps(summary, indent=2))
    return summary


# ----------------------------------------------------------------------------
# Stage A -- BUT ReverbDB scanner -> manifest  (best-effort; VERIFY vs data)
# ----------------------------------------------------------------------------

def _parse_but_meta(meta_path: Path):
    """Parse a BUT ``mic_meta.txt`` (``$Key<TAB>Value``) -> (distance_m, rt60).

    Source-receiver distance is ``$EnvMic{N}RelDistance`` (loudspeaker-to-this-
    microphone-channel distance, verified == ||spk_xyz - mic_xyz||); RT60 is
    ``$EnvMic{N}RelRT60``. Verified on rel_19_06: every mic_meta.txt has exactly
    one such mic block. Returns None if no RelDistance is present/parseable.
    """
    d: dict[str, str] = {}
    for line in meta_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("$"):
            parts = line[1:].split("\t")
            if len(parts) >= 2:
                d[parts[0]] = parts[1].strip()
    dist_keys = [k for k in d if re.fullmatch(r"EnvMic\d+RelDistance", k)]
    if not dist_keys:
        return None
    try:
        dist = float(d[dist_keys[0]])
    except ValueError:
        return None
    rt60 = None
    rt_keys = [k for k in d if re.fullmatch(r"EnvMic\d+RelRT60", k)]
    if rt_keys:
        try:
            rt60 = float(d[rt_keys[0]])
        except ValueError:
            rt60 = None
    return dist, rt60


def scan_but(but_root: str, manifest_out: str) -> int:
    """Scan a downloaded BUT ReverbDB (rel_19_06 RIR-Only) tree into a manifest.

    Layout: ``<Room>/MicID##/SpkID##_<date>_<variant>/<NN>/{mic_meta.txt, RIR/*.wav}``
    where ``<NN>`` is one microphone-ball channel (one RIR). By reciprocity the
    BUT loudspeaker is the bank's receiver, so RIRs are grouped per (room, SpkID)
    -- every microphone channel recorded with the SAME speaker becomes a source
    at some distance to that common receiver. distance = ``$EnvMic*RelDistance``.
    """
    root = Path(but_root)
    if not root.exists():
        raise FileNotFoundError(f"BUT root not found: {root}")
    entries: list[dict] = []
    for meta_path in sorted(root.glob("*/MicID*/SpkID*/*/mic_meta.txt")):
        rirs = sorted(meta_path.parent.glob("RIR/*.wav"))
        if not rirs:
            continue
        parsed = _parse_but_meta(meta_path)
        if parsed is None:
            continue
        dist, rt60 = parsed
        room = meta_path.relative_to(root).parts[0]
        m = re.search(r"(SpkID\d+)", str(meta_path))
        spk = m.group(1) if m else "SpkID"
        entries.append({
            "room_id": f"{room}__{spk}",
            "rir_path": str(rirs[0]),
            "channel": 0,
            "rt60": rt60,
            "distance_m": dist,
        })
    Path(manifest_out).write_text(
        "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries),
        encoding="utf-8",
    )
    n_groups = len({e["room_id"] for e in entries})
    print(f"scan_but: {len(entries)} RIRs across {n_groups} (room,speaker) "
          f"groups -> {manifest_out}")
    if not entries:
        _warn("0 entries -- check the BUT layout/glob under " + str(root))
    return len(entries)


def _warn(msg: str) -> None:
    print(f"[real_rir_to_bank] WARNING: {msg}", file=sys.stderr, flush=True)


# ----------------------------------------------------------------------------
# Self-test -- fabricate a manifest, build a bank, load it with the real
# PreGeneratedRoomBank, and assert the near/far structure round-trips.
# ----------------------------------------------------------------------------

def self_test() -> None:
    import tempfile

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from puresound.audio.rir_bank import PreGeneratedRoomBank

    sr = 16000
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        rir_dir = tmp / "rirs"
        rir_dir.mkdir()
        # Two rooms, each with 2 near (<1m) + 3 far (>=1m) mics; random RIRs.
        manifest = tmp / "manifest.jsonl"
        lines = []

        def _dump(name: str, dist: float, room: str) -> str:
            p = rir_dir / name
            imp = torch.zeros(1, sr)  # 1 s
            imp[0, 10] = 1.0
            imp[0, 200:400] = 0.1 * torch.randn(200)
            torchaudio.save(str(p), imp, sr, encoding="PCM_F", bits_per_sample=32)
            return json.dumps({
                "room_id": room, "rir_path": str(p), "channel": 0,
                "rt60": 0.45, "distance_m": dist,
            })

        for r in range(2):
            for i, dist in enumerate([0.4, 0.8, 2.0, 3.0, 4.5]):
                lines.append(_dump(f"room{r}_mic{i}.wav", dist, f"room{r}"))
        manifest.write_text("\n".join(lines), encoding="utf-8")

        bank_dir = tmp / "bank"
        summary = manifest_to_bank(str(manifest), str(bank_dir), d0=1.0,
                                   target_sr=sr, items_per_room=3, seed=1)
        assert summary["items_written"] == 6, summary
        assert summary["rooms_used"] == 2, summary

        bank = PreGeneratedRoomBank(str(bank_dir), wav_name="rir_5ch.wav",
                                    meta_name="metadata.json")
        assert len(bank) == 6, len(bank)
        scene = bank.sample_scene()
        assert scene["near"] and scene["far"], scene
        assert scene.get("origin") == "real", scene.get("origin")
        for c in scene["near"]:
            assert c["label"] in NEAR_LABELS and c["distance"] < 1.0
        for c in scene["far"]:
            assert c["label"] in FAR_LABELS and c["distance"] >= 1.0
        imp, md, got_sr = bank.select_channel(scene, source_role="foreground")
        assert got_sr == sr and imp.shape[0] == 1
        assert md["label"] in NEAR_LABELS and "drr_db" in md
        imp_f, md_f, _ = bank.select_channel(scene, source_role="interferer")
        assert md_f["label"] in FAR_LABELS

        # near-pool: a far-only room must yield nothing without the pool and
        # borrow pool near channels (<d0) with it.
        fonly = tmp / "faronly.jsonl"
        fonly.write_text("\n".join(
            [_dump(f"fonly_mic{i}.wav", d, "fonly") for i, d in enumerate([1.5, 2.5, 3.5])]
        ), encoding="utf-8")
        pool = tmp / "pool.jsonl"
        pool.write_text("\n".join(
            [_dump(f"pool_mic{i}.wav", d, "poolroom") for i, d in enumerate([0.3, 0.7, 2.2])]
        ), encoding="utf-8")
        s0 = manifest_to_bank(str(fonly), str(tmp / "bank_np0"), d0=1.0,
                              target_sr=sr, items_per_room=3, seed=1)
        assert s0["items_written"] == 0, s0
        s1 = manifest_to_bank(str(fonly), str(tmp / "bank_np1"), d0=1.0,
                              target_sr=sr, items_per_room=3, seed=1,
                              near_pool_manifest=str(pool))
        assert s1["items_written"] == 3, s1
        np_bank = PreGeneratedRoomBank(str(tmp / "bank_np1"),
                                       wav_name="rir_5ch.wav",
                                       meta_name="metadata.json")
        np_scene = np_bank.sample_scene()
        assert np_scene["near"] and np_scene["far"], np_scene
        for c in np_scene["near"]:
            assert c["distance"] < 1.0  # borrowed pool channels only

        print("SELF-TEST OK: emitted items load in PreGeneratedRoomBank; "
              "near/far split + DRR/distance metadata round-trip correctly.")


# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def _write_wav(path: Path, x: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).view(1, -1)
    torchaudio.save(str(path), t, sr, encoding="PCM_F", bits_per_sample=32)


def _onset_sample(x: np.ndarray) -> int:
    """First sample where |x| crosses 1/20 of the global peak (walking back
    from the peak) -- robust direct-path onset for measured RIRs."""
    a = np.abs(x)
    p = int(np.argmax(a))
    thr = a[p] / 20.0
    i = p
    while i > 0 and a[i - 1] > thr:
        i -= 1
    return i


def _append(manifest: Path, entries: list[dict]) -> None:
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("a", encoding="utf-8") as fh:
        for e in entries:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")


# ----------------------------------------------------------------------------
# dEchorate (SOFA re-release)
# ----------------------------------------------------------------------------

def scan_dechorate(input_dir: Path, staging: Path, manifest: Path) -> None:
    import h5py

    pat = re.compile(r"dEchorate_room(\w+)_src(\d+)_arr(\d+)_mics[\d-]+\.sofa$")
    entries: list[dict] = []
    files = sorted(input_dir.glob("*.sofa"))
    for f in files:
        m = pat.search(f.name)
        if not m:
            continue
        code, src, arr = m.group(1), int(m.group(2)), int(m.group(3))
        with h5py.File(f, "r") as h:
            src_pos = np.asarray(h["SourcePosition"][0])          # (3,)
            rec_pos = np.asarray(h["ReceiverPosition"][:, :, 0])  # (5, 3)
            ir = np.asarray(h["Data.IR"][0])                      # (5, N)
            sr = int(np.asarray(h["Data.SamplingRate"])[0])
        for r in range(ir.shape[0]):
            dist = float(np.linalg.norm(src_pos - rec_pos[r]))
            wav = staging / "dechorate" / f"room{code}_src{src}_arr{arr}_r{r}.wav"
            _write_wav(wav, ir[r], sr)
            entries.append({
                "room_id": f"dech_{code}",
                "rir_path": str(wav), "channel": 0,
                "distance_m": round(dist, 4),
            })
    _append(manifest, entries)
    d = np.array([e["distance_m"] for e in entries])
    print(f"dechorate: {len(files)} sofa -> {len(entries)} RIRs; "
          f"<1m {(d < 1).sum()}, 1-5m {((d >= 1) & (d <= 5)).sum()}, "
          f"dmin {d.min():.2f} dmax {d.max():.2f}")


# ----------------------------------------------------------------------------
# BRUDEX (nominal geometry + TOA sanity fit)
# ----------------------------------------------------------------------------

_BRUDEX_RT60 = {"low": 0.31, "medium": 0.51, "high": 1.30}
_BRUDEX_HEAD = np.array([2.90, 3.49])


def _brudex_emic_xy(n: int) -> np.ndarray:
    """E{n} (1-based) -> planar position; row-major from bottom-left E1."""
    i, j = (n - 1) % 6, (n - 1) // 6
    return np.array([0.75 + 0.95 * i, 0.80 + 0.95 * j])


def _brudex_src_xy(theta_deg: float) -> np.ndarray:
    t = math.radians(theta_deg)
    return _BRUDEX_HEAD + 2.0 * np.array([math.sin(t), math.cos(t)])


def scan_brudex(input_dir: Path, staging: Path, manifest: Path) -> None:
    import h5py

    pat = re.compile(r"RIR_(low|medium|high)_eMicRun_(\d)_DOA_(-?\d+)deg\.mat$")
    entries: list[dict] = []
    dropped = 0
    for f in sorted(input_dir.rglob("RIR_*_eMicRun_*.mat")):
        m = pat.search(f.name)
        if not m:
            continue
        cond, run, theta = m.group(1), int(m.group(2)), int(m.group(3))
        with h5py.File(f, "r") as h:
            keys = [k for k in h.keys() if not k.startswith("#")]
            ds = max((h[k] for k in keys if hasattr(h[k], "shape")),
                     key=lambda d: int(np.prod(d.shape)))
            rir = np.asarray(ds)  # v7.3 stores transposed: expect (18, N) or (N, 18)
        if rir.ndim != 2:
            raise ValueError(f"{f.name}: unexpected shape {rir.shape}")
        if rir.shape[0] > rir.shape[1]:
            rir = rir.T  # -> (18, N)
        if rir.shape[0] != 18:
            raise ValueError(f"{f.name}: expected 18 channels, got {rir.shape}")
        src = _brudex_src_xy(theta)
        for ch in range(6, 18):
            e_idx = (run - 1) * 12 + (ch - 6) + 1
            nominal = float(np.linalg.norm(src - _brudex_emic_xy(e_idx)))
            toa = _onset_sample(rir[ch]) / 48000.0 * 343.0
            if abs(toa - nominal) > 0.5:
                dropped += 1
                continue
            wav = staging / "brudex" / f"{cond}_DOA{theta}_E{e_idx:02d}.wav"
            _write_wav(wav, rir[ch], 48000)
            entries.append({
                "room_id": f"brudex_{cond}",
                "rir_path": str(wav), "channel": 0,
                "distance_m": round(toa, 4),
                "rt60": _BRUDEX_RT60[cond],
            })
    _append(manifest, entries)
    d = np.array([e["distance_m"] for e in entries])
    print(f"brudex: {len(entries)} RIRs kept, {dropped} dropped "
          f"(|TOA-nominal|>0.5m); <1m {(d < 1).sum()}, "
          f"1-5m {((d >= 1) & (d <= 5)).sum()}, dmin {d.min():.2f} dmax {d.max():.2f}")


# ----------------------------------------------------------------------------
# openSLR-28 REVERB real RIRs (direct wavs, documented distances)
# ----------------------------------------------------------------------------

_REVERB_RT60 = {"smallroom": 0.25, "mediumroom": 0.50, "largeroom": 0.75}


def scan_slr28(input_dir: Path, staging: Path, manifest: Path) -> None:
    del staging  # wavs used in place (8-channel, addressed via manifest channel)
    pat = re.compile(
        r"RVB2014_type1_rir_(smallroom|mediumroom|largeroom)(\d)_(near|far)_angl([ab])\.wav$"
    )
    entries: list[dict] = []
    for f in sorted(input_dir.rglob("RVB2014_type1_rir_*.wav")):
        m = pat.search(f.name)
        if not m:
            continue
        size, idx, pos, _ang = m.groups()
        info = torchaudio.info(str(f))
        for ch in range(info.num_channels):
            entries.append({
                "room_id": f"reverb_{size}{idx}",
                "rir_path": str(f), "channel": ch,
                "distance_m": 0.5 if pos == "near" else 2.0,
                "rt60": _REVERB_RT60[size],
            })
    _append(manifest, entries)
    d = np.array([e["distance_m"] for e in entries])
    print(f"slr28: {len(entries)} RIR channels; <1m {(d < 1).sum()}, "
          f">=1m {(d >= 1).sum()}")


# ----------------------------------------------------------------------------
# ACE Single (distance from direct-path TOA)
# ----------------------------------------------------------------------------

_ACE_RT60 = {  # full-band T60 means, ACE corpus paper (Eaton et al., 2015)
    "Office_1": 0.332, "Office_2": 0.390,
    "Meeting_Room_1": 0.437, "Meeting_Room_2": 0.371,
    "Lecture_Room_1": 0.638, "Lecture_Room_2": 1.220,
    "Building_Lobby": 0.646,
}


# source-mic distance (m), ACE tech report arXiv 1606.03365 Table 4,
# "Ch-1 of 8-ch. lin." column (= the Single config per the corpus paper).
_ACE_DIST = {
    ("Office_1", "1"): 1.16, ("Office_1", "2"): 2.68,
    ("Office_2", "1"): 1.33, ("Office_2", "2"): 2.43,
    ("Meeting_Room_1", "1"): 1.35, ("Meeting_Room_1", "2"): 2.56,
    ("Meeting_Room_2", "1"): 1.80, ("Meeting_Room_2", "2"): 2.75,
    ("Lecture_Room_1", "1"): 1.20, ("Lecture_Room_1", "2"): 2.72,
    ("Lecture_Room_2", "1"): 1.93, ("Lecture_Room_2", "2"): 2.70,
    ("Building_Lobby", "1"): 1.45, ("Building_Lobby", "2"): 2.95,
}


def scan_ace(input_dir: Path, staging: Path, manifest: Path) -> None:
    del staging
    entries: list[dict] = []
    for f in sorted(input_dir.rglob("Single_*_RIR.wav")):
        room = f.parent.parent.name
        pos = f.parent.name
        dist = _ACE_DIST.get((room, pos))
        if dist is None:
            print(f"  ace: {room}/{pos} not in distance table, skipped")
            continue
        print(f"  ace {room}/{pos}: distance {dist:.2f} m")
        entries.append({
            "room_id": f"ace_{room}",
            "rir_path": str(f), "channel": 0,
            "distance_m": dist,
            "rt60": _ACE_RT60.get(room),
        })
    _append(manifest, entries)
    print(f"ace: {len(entries)} RIRs (all >=1m; near comes from --near-pool)")


# ----------------------------------------------------------------------------
# DIFFRIR (Hearing Anything Anywhere)
# ----------------------------------------------------------------------------

# speaker_xyz per config, from github.com/maswang32/hearinganythinganywhere
# rooms/*.py (TOA-calibrated; comments there note 4-51 cm residual error).
_DIFFRIR_SPEAKER = {
    "classroomBase": [3.5838, 5.7230, 1.2294],
    "dampenedBase": [2.4542, 2.4981, 1.2654],
    "dampenedRotation": [2.4595, 2.6748, 1.0659],
    "dampenedTranslation": [1.2621, 0.5605, 1.2404],
    "dampenedPanel": [2.4052, 2.5292, 1.3726],
    "hallwayBase": [0.6870, 10.2452, 0.5367],
    "hallwayPanel1": [0.5091, 10.4333, 0.5464],
    "hallwayPanel2": [1.2120, 17.1969, 0.5444],
    "hallwayPanel3": [0.6549, 10.2356, 0.4618],
    "hallwayRotation": [0.5002, 10.1438, 0.3348],
    "hallwayTranslation": [0.4746, 9.9688, 0.2613],
    "complexBase": [2.8377, 10.1228, 1.1539],
}


def scan_diffrir(input_dir: Path, staging: Path, manifest: Path) -> None:
    entries: list[dict] = []
    for zpath in sorted(input_dir.glob("*.zip")):
        cfg = zpath.stem
        if cfg not in _DIFFRIR_SPEAKER:
            print(f"  diffrir: skipping {cfg} (no speaker_xyz known)")
            continue
        spk = np.array(_DIFFRIR_SPEAKER[cfg])
        with zipfile.ZipFile(zpath) as z:
            names = z.namelist()
            xyz_name = next(n for n in names if n.endswith("/xyzs.npy"))
            rir_name = next(
                n for n in names
                if n.rsplit("/", 1)[-1].lower() in ("rirs.npy", "rir.npy")
            )
            with z.open(xyz_name) as fh:
                xyz = np.load(fh)
            with z.open(rir_name) as fh:
                rirs = np.load(fh)  # (M, N) float
        if rirs.shape[0] != xyz.shape[0]:
            raise ValueError(f"{cfg}: RIRs {rirs.shape} vs xyzs {xyz.shape}")
        room_key = re.match(r"[a-z]+", cfg).group(0)
        for i in range(xyz.shape[0]):
            dist = float(np.linalg.norm(xyz[i] - spk))
            wav = staging / "diffrir" / cfg / f"{cfg}_{i:04d}.wav"
            _write_wav(wav, rirs[i].astype(np.float32), 48000)
            entries.append({
                # each config = its own acoustic room (panels move), but keep
                # the physical room in the id for held-out splitting
                "room_id": f"diffrir_{room_key}_{cfg}",
                "rir_path": str(wav), "channel": 0,
                "distance_m": round(dist, 4),
            })
        print(f"  diffrir {cfg}: {xyz.shape[0]} RIRs")
    _append(manifest, entries)
    d = np.array([e["distance_m"] for e in entries])
    print(f"diffrir: {len(entries)} RIRs; <1m {(d < 1).sum()}, "
          f"1-5m {((d >= 1) & (d <= 5)).sum()}, dmin {d.min():.2f}")


# ----------------------------------------------------------------------------

SCANNERS = {
    "dechorate": scan_dechorate,
    "brudex": scan_brudex,
    "slr28": scan_slr28,
    "ace": scan_ace,
    "diffrir": scan_diffrir,
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("from-manifest", help="Stage B: manifest.jsonl -> bank")
    b.add_argument("--manifest", required=True)
    b.add_argument("--output", required=True)
    b.add_argument("--d0", type=float, default=1.0, help="near/far split distance (m)")
    b.add_argument("--target-sr", type=int, default=16000)
    b.add_argument("--items-per-room", type=int, default=4)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--near-pool", default=None,
                   help="manifest.jsonl of real <d0 RIRs; rooms with far but "
                        "no near entries borrow their near channels from it")

    t = sub.add_parser("but", help="Stage A+B: scan BUT ReverbDB -> bank")
    t.add_argument("--input", required=True, help="BUT ReverbDB root")
    t.add_argument("--output", required=True)
    t.add_argument("--manifest", default=None, help="manifest path (default <output>/manifest.jsonl)")
    t.add_argument("--d0", type=float, default=1.0)
    t.add_argument("--target-sr", type=int, default=16000)
    t.add_argument("--items-per-room", type=int, default=4)
    t.add_argument("--seed", type=int, default=0)

    s = sub.add_parser("scan", help="Stage A: scan a public measured-RIR corpus -> manifest")
    s.add_argument("corpus", choices=sorted(SCANNERS))
    s.add_argument("--input", required=True, type=Path, help="downloaded corpus directory")
    s.add_argument("--staging", required=True, type=Path,
                   help="where to dump mono wavs when the source is HDF5/MAT/npy")
    s.add_argument("--manifest", required=True, type=Path, help="manifest.jsonl to write")

    sub.add_parser("self-test", help="verify Stage B against PreGeneratedRoomBank")

    args = p.parse_args()
    if args.cmd == "scan":
        if args.manifest.exists():
            args.manifest.unlink()  # scanners append; start fresh per invocation
        SCANNERS[args.corpus](args.input, args.staging, args.manifest)
    elif args.cmd == "self-test":
        self_test()
    elif args.cmd == "from-manifest":
        manifest_to_bank(args.manifest, args.output, args.d0, args.target_sr,
                         args.items_per_room, args.seed,
                         near_pool_manifest=args.near_pool)
    elif args.cmd == "but":
        manifest = args.manifest or str(Path(args.output) / "manifest.jsonl")
        Path(args.output).mkdir(parents=True, exist_ok=True)
        n = scan_but(args.input, manifest)
        if n:
            manifest_to_bank(manifest, args.output, args.d0, args.target_sr,
                             args.items_per_room, args.seed)


if __name__ == "__main__":
    main()
