"""A level-matched copy of the field set, to separate calibration from representation.

The QVF publication clips run ~14 dB hotter than our own device recordings, and
the model's near/far decision is partly absolute-level driven (rho(in_dbfs,
pred_fg_dist) = -0.49). So a genuinely distant talker in a hot recording reads as
a near user. This applies ONE per-recording gain so every recording's own user
voice sits at the device median, changing the absolute scale and nothing inside
the recording -- the near/far relationship, the SNR and the reverberation are all
preserved.

Anchor is `near_ref_dbfs`, the energy of the recording's keep spans. A recording
with no keep span (qvf_scenario2, lone far only) has no user voice to anchor on
and falls back to the noise floor; that clip is reported separately because a
floor anchor is a different normalisation, not the same one.
"""
import json, pathlib, shutil, sys
import numpy as np
import soundfile as sf

SRC = pathlib.Path("data_report/field_cases/test_vector_cases")
DST = pathlib.Path(sys.argv[1])
w = json.load(open(SRC / "windows.json"))

dev_nr = [v["near_ref_dbfs"] for k, v in w.items()
          if not k.startswith(("_", "qvf")) and v.get("near_ref_dbfs") == v.get("near_ref_dbfs")]
dev_fl = [v["floor_dbfs"] for k, v in w.items()
          if not k.startswith(("_", "qvf")) and v.get("floor_dbfs") is not None]
TARGET_NR, TARGET_FL = float(np.median(dev_nr)), float(np.median(dev_fl))
print(f"target: near_ref {TARGET_NR:.2f} dBFS   (floor fallback {TARGET_FL:.2f})")

# one gain per RECORDING (group), from its own reference block
gain_db, anchor = {}, {}
for k, v in w.items():
    if k.startswith("_"):
        continue
    g = v["group"]
    if g in gain_db:
        continue
    nr = v.get("near_ref_dbfs")
    if nr is not None and nr == nr:
        gain_db[g] = TARGET_NR - float(nr); anchor[g] = "near_ref"
    else:
        gain_db[g] = TARGET_FL - float(v["floor_dbfs"]); anchor[g] = "floor"
for g in sorted(gain_db, key=lambda x: gain_db[x]):
    print(f"  {g:20s} {gain_db[g]:+7.2f} dB   (anchor: {anchor[g]})")

DST.mkdir(parents=True, exist_ok=True)
out = {"_comment": w["_comment"] + (
    " LEVEL-MATCHED COPY: one gain per recording so its near_ref_dbfs sits at the "
    f"device median {TARGET_NR:.2f} dBFS. Absolute scale only -- nothing inside a "
    "recording changed. Built by renorm.py to separate level calibration from "
    "representation; NOT a benchmark set.")}
clipped = 0
for k, v in w.items():
    if k.startswith("_"):
        continue
    g = gain_db[v["group"]]
    lin = 10.0 ** (g / 20.0)
    for suffix in ("_raw", "_qvf22"):
        p = SRC / f"{k}{suffix}.wav"
        if not p.is_file():
            continue
        x, sr = sf.read(p)
        y = x * lin
        if np.abs(y).max() > 1.0:
            clipped += 1
            y = y / np.abs(y).max() * 0.999
        sf.write(DST / f"{k}{suffix}.wav", y.astype(np.float32), sr)
    spec = dict(v)
    for key in ("in_dbfs", "near_ref_dbfs", "floor_dbfs"):
        if spec.get(key) is not None and spec[key] == spec[key]:
            spec[key] = round(float(spec[key]) + g, 2)
    spec["level_gain_db"] = round(g, 2)
    spec["level_anchor"] = anchor[v["group"]]
    out[k] = spec
(DST / "windows.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
print(f"\nwrote {len([k for k in out if not k.startswith('_')])} entries to {DST}")
print(f"clip-limited files: {clipped}" if clipped else "no clipping")
