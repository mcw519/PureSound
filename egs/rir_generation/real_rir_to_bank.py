"""Convert a REAL measured RIR dataset into a PreGeneratedRoomBank folder.

Simulated RIRs leave a gap between in-domain separation and real-recording
performance. Measured RIRs with a known source-receiver DISTANCE let a bank be
built whose near/far structure matches the training task while its acoustics come
from real rooms, which closes the RIR part of that gap.

Two stages, decoupled by a manifest so the (definitive, self-tested) bank-format
emission is independent of any one dataset's on-disk layout:

  Stage A  dataset -> manifest.jsonl   (thin, dataset-specific scanner)
  Stage B  manifest.jsonl -> bank/     (THIS is the part that must be exactly
                                        right; it is self-tested against
                                        PreGeneratedRoomBank)

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
from pathlib import Path

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

    sub.add_parser("self-test", help="verify Stage B against PreGeneratedRoomBank")

    args = p.parse_args()
    if args.cmd == "self-test":
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
