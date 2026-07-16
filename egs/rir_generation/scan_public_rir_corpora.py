"""Stage-A scanners: public measured-RIR corpora -> real_rir_to_bank manifests.

Each scanner walks one downloaded corpus, extracts every usable mono RIR
(dumping a staging wav when the source is HDF5/MAT/npy), computes the
source-microphone DISTANCE from the corpus' own geometry, and appends
manifest.jsonl lines consumed by ``real_rir_to_bank.py from-manifest``
(the definitive, self-tested Stage B). Corpora covered:

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

Usage:
    uv run python egs/rir_generation/scan_public_rir_corpora.py <corpus> \
        --input <corpus dir> --staging <wav dump dir> --manifest <out.jsonl>
"""
from __future__ import annotations

import argparse
import json
import math
import re
import zipfile
from pathlib import Path

import numpy as np
import torch
import torchaudio


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
    p.add_argument("corpus", choices=sorted(SCANNERS))
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--staging", required=True, type=Path)
    p.add_argument("--manifest", required=True, type=Path)
    args = p.parse_args()
    if args.manifest.exists():
        args.manifest.unlink()  # scanners append; start fresh per invocation
    SCANNERS[args.corpus](args.input, args.staging, args.manifest)


if __name__ == "__main__":
    main()
