"""Step 1 of the anchor-qualification gate: cache what an inference-side supervisor could read.

For every field cold-start clip under the context conditions of `reference_matrix_probe`
(none / floor / near_speech / far_speech / event / near_xchain / stream), every session clip
(none), and every Dawn Chorus utterance under `none` and `background` context, run the model
once and store, per (ckpt, clip, condition):

  mix        float16 [T]        model input INCLUDING the context prefix
  enh        float16 [T]        model output at dry_blend 1.0
  feat       float16 [C, F]     bottleneck pooled over frequency, per frame (100 fps)
  dist       float32 [F, 3]     DistHead applied per frame to `feat` -- [drr/10, log10 fg_m, log10 itf_m]
                                (the head is mean(F,T)->MLP; a sliding-window estimate is
                                 MLP(mean over the window of `feat`), computed offline)
  offset_s   float              where the clip proper starts (prefix length)
  spans      json               keep / suppress spans in CLIP time (add offset_s)
  side, group, floor_dbfs, role

Requires the TRAINING config (it builds the dist_head; infer_dpcrn.yaml does not). Output:
one .npz per (clip, condition) under <out>/<ckpt_tag>/field/ and <out>/<ckpt_tag>/dawn/, plus
an index.jsonl per ckpt. Step 2 (offline gate simulation) reads only these files.

  uv run python benchmarks/probes/anchor_gate_cache.py field config/train_dpcrn.yaml \
      --ckpt pretrained_ckpt/dpcrn_v8.ckpt --tag v8 --out <cache_dir> --device cuda:0
  uv run python benchmarks/probes/anchor_gate_cache.py dawn  config/train_dpcrn.yaml \
      --ckpt ... --tag v8 --out <cache_dir> --device cuda:1 [--limit N]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
RECIPE_DIR = REPO_ROOT / "egs/voice_isolate"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(RECIPE_DIR / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from puresound.config import load_recipe         # noqa: E402
from puresound.recipes import init_siso_model    # noqa: E402
import reference_matrix_probe as rm              # noqa: E402

SR = 16000
FPS = 100.0


def load_model(config_path: str, ckpt: str, device: torch.device, head: str = "dist"):
    model = init_siso_model(load_recipe(config_path, expected_task="voice_isolation",
                                        expected_purpose="train").model)
    state = torch.load(ckpt, map_location="cpu")
    if head == "proximity":
        from puresound.evaluation.session_validation import require_checkpoint_heads
        require_checkpoint_heads(model, state.get("state_dict", state), heads=("proximity_head",))
    model.reload_checkpoint(state.get("state_dict", state), load_loss_func=False)
    model = model.to(device).eval()
    if getattr(model.backbone, f"{head}_head", None) is None:
        raise SystemExit(f"config builds no {head}_head -- pass its TRAINING config")
    model._anchor_cache_head = head
    model.backbone.stash_bottleneck = True
    return model


def save_head(model, out: Path) -> None:
    """The dist_head MLP weights, so step 2 can evaluate MLP(mean over a window of `feat`)
    exactly instead of averaging per-frame readouts."""
    out.mkdir(parents=True, exist_ok=True)
    if getattr(model, "_anchor_cache_head", "dist") == "proximity":
        (out / "proximity_head.json").write_text(json.dumps({"head": "proximity", "units": "raw scalar", "near_direction": "higher"}))
        return
    torch.save({k: v.cpu() for k, v in model.backbone.dist_head.net.state_dict().items()},
               out / "dist_head_net.pt")


@torch.no_grad()
def run(model, x: torch.Tensor, device) -> dict:
    """x: [1, T] float. Returns enh [T], feat [C, F], dist [F, 3] (per-frame head)."""
    out = model(x.to(device)).detach().cpu().view(-1)
    bott = model.backbone.last_bottleneck            # [1, C, Fbins, T]
    feat = bott.mean(dim=2)[0]                       # [C, T] pooled over frequency
    vad = getattr(model.backbone, "last_vad_logits", None)
    rec = {"enh": out.numpy().astype(np.float16),
           "feat": feat.detach().cpu().numpy().astype(np.float16)}
    if getattr(model, "_anchor_cache_head", "dist") == "proximity":
        proximity = getattr(model.backbone, "last_proximity", None)
        if proximity is None:
            raise ValueError("ProximityHead did not emit last_proximity")
        rec["proximity"] = proximity.detach().cpu().reshape(-1).numpy().astype(np.float32)
    else:
        dist = model.backbone.dist_head.net(feat.transpose(0, 1))
        rec["dist"] = dist.detach().cpu().numpy().astype(np.float32)
    if vad is not None:
        rec["vad_logit"] = vad.detach().cpu().reshape(-1).numpy().astype(np.float32)
    return rec


def save(path: Path, x: torch.Tensor, rec: dict, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, mix=x.view(-1).numpy().astype(np.float16), meta=json.dumps(meta), **rec)


def cmd_field(args) -> None:
    device = torch.device(args.device)
    model = load_model(args.config_path, args.ckpt, device, args.head)
    save_head(model, Path(args.out) / args.tag)
    windows = {k: v for k, v in json.loads((rm.CASES / "windows.json").read_text()).items()
               if not k.startswith("_")}
    out_dir = Path(args.out) / args.tag / "field"
    index = []
    for clip, spec in windows.items():
        role = spec.get("role", "")
        if clip.endswith("_session"):
            side, conds = "session", [("none", "none", None, {})]
            spans = {"keep": spec.get("keep") or [], "suppress": spec.get("suppress") or []}
        else:
            if "lone far" in role:
                side = "far"
            elif "lone near" in role:
                side = "near"
            elif "double-talk" in role:
                side = "dt"
            else:
                continue
            conds = rm.conditions_for(clip, spec, windows, sweep=False)
            spans = {"keep": spec.get("keep") or [], "suppress": spec.get("suppress") or []}
        wav = rm.load_mono(rm.CASES / f"{clip}_raw.wav").view(1, -1)
        for cond, variant, pad, meta in conds:
            x = torch.cat([pad.view(1, -1), wav], dim=-1) if pad is not None else wav
            off = (pad.numel() / SR) if pad is not None else 0.0
            rec = run(model, x, device)
            m = {"clip": clip, "group": rm.group_of(clip), "side": side, "condition": cond,
                 "variant": variant, "offset_s": round(off, 3), "spans": spans,
                 "floor_dbfs": spec.get("floor_dbfs"), "near_ref_dbfs": spec.get("near_ref_dbfs"),
                 "held_out": bool(spec.get("held_out")), "sentinel": bool(spec.get("sentinel")),
                 "fps": FPS, "sr": SR, "ckpt": args.ckpt, **{k: v for k, v in meta.items() if k != "content"}}
            if "content" in meta:
                m["stream_content"] = meta["content"]
            fn = out_dir / f"{clip}__{variant}.npz"
            save(fn, x, rec, m)
            index.append({**m, "file": str(fn)})
        print(f"  {clip}: {len(conds)} conditions", flush=True)
    (Path(args.out) / args.tag / "field_index.jsonl").write_text(
        "\n".join(json.dumps(r) for r in index) + "\n")
    print(f"wrote {len(index)} field records under {out_dir}")


def cmd_dawn(args) -> None:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from eval_dawn_chorus import DATASET_REPO, load_wav_bytes, context_input

    device = torch.device(args.device)
    model = load_model(args.config_path, args.ckpt, device, args.head)
    save_head(model, Path(args.out) / args.tag)
    table = pq.read_table(hf_hub_download(DATASET_REPO, "eval.parquet", repo_type="dataset"))
    n = min(args.limit, table.num_rows) if args.limit else table.num_rows
    out_dir = Path(args.out) / args.tag / "dawn"
    index, fallback = [], 0
    for i in range(n):
        item = table.slice(i, 1).to_pylist()[0]
        mix = load_wav_bytes(item["mix"]["bytes"], SR)
        ref = load_wav_bytes(item["speech"]["bytes"], SR)
        L = min(len(mix), len(ref))
        mix, ref = mix[:L], ref[:L]
        for ctx in ("none", "background"):
            model_in, offset, fell = context_input(mix, ref, SR, ctx)
            fallback += int(fell and ctx == "background")
            x = torch.from_numpy(model_in.astype(np.float32)).view(1, -1)
            rec = run(model, x, device)
            m = {"id": item["id"], "condition": ctx, "offset_s": offset / SR, "fell_back": bool(fell),
                 "transcript": item["transcript"], "conversation_type": item["conversation_type"],
                 "fps": FPS, "sr": SR, "ckpt": args.ckpt}
            fn = out_dir / f"{item['id']}__{ctx}.npz"
            rec["ref"] = ref.astype(np.float16)
            save(fn, x, rec, m)
            index.append({**m, "file": str(fn)})
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{n}", flush=True)
    (Path(args.out) / args.tag / "dawn_index.jsonl").write_text(
        "\n".join(json.dumps(r) for r in index) + "\n")
    print(f"wrote {len(index)} dawn records under {out_dir}; background fallbacks: {fallback}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("field", cmd_field), ("dawn", cmd_dawn)):
        s = sub.add_parser(name)
        s.add_argument("config_path"); s.add_argument("--ckpt", required=True)
        s.add_argument("--tag", required=True); s.add_argument("--out", required=True)
        s.add_argument("--head", choices=["dist", "proximity"], default="dist")
        s.add_argument("--device", default="cuda"); s.add_argument("--limit", type=int, default=None)
        s.set_defaults(fn=fn)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
