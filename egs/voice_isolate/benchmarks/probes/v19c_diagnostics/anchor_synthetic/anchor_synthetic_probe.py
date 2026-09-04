"""Does the anchor behaviour of reference_matrix_README §2-§3 reproduce on SYNTHETIC data?

Conditions are built from data_report/wer_set_moderate_test (200 utts, mix = ref + interferers
+ noise; ref = near-reverb foreground). Every condition is a PREFIX prepended to the utterance's
mix; the model runs once over prefix+utterance and only the utterance part is scored, paired
against the `none` condition per utterance.

prefixes
  none                      -- the utterance alone
  silence3                  -- 3 s of digital zeros
  bg3                       -- 3 s of the mix's ref-INACTIVE 20 ms frames, in time order, tiled
                               (== eval_dawn_chorus `background`; needs >= 0.3 s of them)
  bgsub3                    -- 3 s of (mix - ref), i.e. interferers+noise with the user removed
                               (always available; no fallback)
  self3                     -- 3 s of the SAME utterance's ref-ACTIVE mix frames (same talker)
  other{L}, L in .25/.5/1/2/3 -- L s of ANOTHER utterance's ref-ACTIVE mix frames (different spk)
  decaybg{N}, N in 2/5/10/20  -- other3, then N s of that utterance's own bg3 material, tiled
  decayq{N},  N in 2/5/10/20  -- other3, then N s of the QUIETEST 30% of (mix - ref) frames,
                               tiled (the synthetic analogue of "true floor, same chain")

metrics on the utterance part only (20 ms frames, ref-active = frame energy >= max-40 dB and
>= -60 dBFS)
  del_db   = 10log10( mean out^2 / mean mix^2 ) over ref-ACTIVE frames   (more negative = more
             of the user removed -- the deletion proxy)
  leak_db  = same over ref-INACTIVE frames                              (suppression of itf+noise)
  sisdr    = SI-SDR(out, ref);  sisdri = sisdr - SI-SDR(mix, ref)

usage (from egs/voice_isolate)
  uv run python <this> run config/train_dpcrn.yaml --ckpt pretrained_ckpt/dpcrn_v8.ckpt \
      --tag v8 --out <dir> --device cuda:0
  uv run python <this> report <dir>/rows_v8.jsonl <dir>/rows_v16ep19.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path("/home/milowu/A4Audio/PureSound")
RECIPE_DIR = REPO_ROOT / "egs/voice_isolate"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(RECIPE_DIR / "scripts"))

SR = 16000
FRAME = 320          # 20 ms
REL_DB = 40.0
ABS_DBFS = -60.0
BG_MIN_SEC = 0.3
SWEEP = [0.25, 0.5, 1.0, 2.0, 3.0]
GAPS = [2.0, 5.0, 10.0, 20.0]


# ---------------------------------------------------------------- frames / masks
def frame_energy(x: np.ndarray, n_frames: int) -> np.ndarray:
    return (x[: n_frames * FRAME].reshape(n_frames, FRAME).astype(np.float64) ** 2).mean(axis=1)


def active_mask(ref: np.ndarray, n_frames: int) -> np.ndarray:
    e = frame_energy(ref, n_frames)
    inactive = (e < e.max() * 10.0 ** (-REL_DB / 10.0)) | (e < 10.0 ** (ABS_DBFS / 10.0))
    return ~inactive


def tile_to(seg: np.ndarray, sec: float) -> np.ndarray:
    need = int(round(sec * SR))
    if len(seg) == 0:
        return np.zeros(need, dtype=np.float32)
    if len(seg) < need:
        seg = np.tile(seg, int(np.ceil(need / len(seg))))
    return seg[:need].astype(np.float32)


def frames_where(src: np.ndarray, mask: np.ndarray) -> np.ndarray:
    nf = len(mask)
    return src[: nf * FRAME].reshape(nf, FRAME)[mask].reshape(-1)


def onset_masks(act: np.ndarray) -> dict[str, np.ndarray]:
    """Active frames inside the first 1 s / 2 s of clip time after the talker's first active
    frame, and the active frames after 2 s. Wrong-anchor / onset deletion should live in `on1`."""
    idx = np.flatnonzero(act)
    if len(idx) == 0:
        return {}
    f0 = idx[0]
    ar = np.arange(len(act))
    return {"on1": act & (ar < f0 + 50), "on2": act & (ar < f0 + 100),
            "late": act & (ar >= f0 + 100)}


# ---------------------------------------------------------------- metrics
def db_ratio(out: np.ndarray, mix: np.ndarray, mask: np.ndarray) -> float | None:
    if mask.sum() == 0:
        return None
    nf = len(mask)
    eo = frame_energy(out, nf)[mask].mean()
    ei = frame_energy(mix, nf)[mask].mean()
    return float(10.0 * np.log10((eo + 1e-14) / (ei + 1e-14)))


def si_sdr(est: np.ndarray, ref: np.ndarray, eps: float = 1e-8) -> float:
    est = est.astype(np.float64); ref = ref.astype(np.float64)
    est = est - est.mean(); ref = ref - ref.mean()
    a = (est @ ref) / ((ref @ ref) + eps)
    s = a * ref; e = est - s
    return float(10.0 * np.log10(((s @ s) + eps) / ((e @ e) + eps)))


# ---------------------------------------------------------------- data
def load_set(set_dir: Path, limit: int | None):
    import soundfile as sf
    items = [json.loads(l) for l in open(set_dir / "manifest.jsonl", encoding="utf-8")]
    if limit:
        items = items[:limit]
    out = []
    for it in items:
        mix, sr1 = sf.read(set_dir / f"{it['id']}_mix.wav", dtype="float32", always_2d=False)
        ref, sr2 = sf.read(set_dir / f"{it['id']}_ref.wav", dtype="float32", always_2d=False)
        assert sr1 == sr2 == SR, (sr1, sr2)
        if mix.ndim > 1:
            mix = mix.mean(1)
        if ref.ndim > 1:
            ref = ref.mean(1)
        L = min(len(mix), len(ref))
        out.append({**it, "mix": np.ascontiguousarray(mix[:L]), "ref": np.ascontiguousarray(ref[:L])})
    return out


def build_material(rec: dict) -> dict:
    """Per-utterance prefix material, computed once."""
    mix, ref = rec["mix"], rec["ref"]
    nf = min(len(mix), len(ref)) // FRAME
    act = active_mask(ref, nf)
    bg_sub = (mix[: nf * FRAME] - ref[: nf * FRAME]).astype(np.float32)
    # quietest 30% of the background-only frames, kept in time order
    e = frame_energy(bg_sub, nf)
    k = max(1, int(0.3 * nf))
    thr = np.sort(e)[k - 1]
    quiet = frames_where(bg_sub, e <= thr)
    bg_inact = frames_where(mix, ~act)
    return {
        "nf": nf, "act": act,
        "speech": frames_where(mix, act),              # ref-active mix frames (the talker)
        "bg_inact": bg_inact if len(bg_inact) >= int(BG_MIN_SEC * SR) else None,
        "bg_sub": bg_sub,
        "quiet": quiet,
    }


def partner_index(items: list) -> list[int]:
    """A different utterance with a DIFFERENT speaker, deterministic."""
    n = len(items)
    out = []
    for i in range(n):
        j = (i + n // 2) % n
        tries = 0
        while (j == i or str(items[j]["spk"]) == str(items[i]["spk"])) and tries < n:
            j = (j + 1) % n
            tries += 1
        out.append(j)
    return out


def same_spk_index(items: list) -> list[int | None]:
    """A DIFFERENT utterance by the SAME speaker (different room draw), or None."""
    import collections
    by = collections.defaultdict(list)
    for k, it in enumerate(items):
        by[str(it["spk"])].append(k)
    out: list[int | None] = []
    for i, it in enumerate(items):
        pool = [k for k in by[str(it["spk"])] if k != i]
        out.append(pool[i % len(pool)] if pool else None)
    return out


def conditions_ident(i: int, mat: list, part: list, same: list) -> list:
    """Is the wrong-anchor deletion keyed on SPEAKER IDENTITY or on the ACOUSTIC/CHANNEL
    signature? `other` differs in both (new talker, new room draw), so on its own it cannot say.

      self{L}     same talker, same utterance, same room   -- the null in the utt arm
      selflate{L} same talker, same utterance, same room, a LATER excerpt (controls for the
                  anchor being byte-identical to the clip's own opening)
      samespk{L}  SAME talker, different utterance => DIFFERENT room / distance / level
      diffspk{L}  different talker, different room            (== `other{L}`)
    """
    m, o = mat[i], mat[part[i]]
    s = mat[same[i]] if same[i] is not None else None
    conds: list[tuple[str, np.ndarray | None]] = [("none", None)]
    for L in (1.0, 3.0):
        sp = m["speech"]
        conds.append((f"self{L:g}", tile_to(sp, L)))
        conds.append((f"selflate{L:g}", tile_to(sp[-int(L * SR):] if len(sp) > int(L * SR)
                                                else sp, L)))
        conds.append((f"samespk{L:g}", None if s is None else tile_to(s["speech"], L)))
        conds.append((f"diffspk{L:g}", tile_to(o["speech"], L)))
    return conds


def conditions(i: int, mat: list, part: list, arm: str) -> list[tuple[str, np.ndarray | None]]:
    """arm 'utt': anchors are OTHER utterances' talkers, the scored signal is the user's mix.
    arm 'bgonly': the scored signal is (mix - ref) -- interferers+noise with NO user, the
    synthetic analogue of a lone-far field clip; `self3` is then the user's OWN near speech
    (the anchor that should unlock suppression of the interferer)."""
    m, o = mat[i], mat[part[i]]
    conds: list[tuple[str, np.ndarray | None]] = [("none", None)]
    conds.append(("silence3", np.zeros(3 * SR, dtype=np.float32)))
    conds.append(("bg3", tile_to(m["bg_inact"], 3.0) if m["bg_inact"] is not None else None))
    conds.append(("bgsub3", tile_to(m["bg_sub"], 3.0)))
    conds.append(("quiet3", tile_to(m["quiet"], 3.0)))
    conds.append(("self3", tile_to(m["speech"], 3.0)))
    for L in SWEEP:
        conds.append((f"other{L:g}", tile_to(o["speech"], L)))
    if arm == "bgonly":
        # anchor-length sweep on the utterance's OWN near talker (the deployment case:
        # the user speaks, then a lone interferer continues)
        for L in SWEEP:
            conds.append((f"self{L:g}", tile_to(m["speech"], L)))
        a3 = tile_to(m["speech"], 3.0)
    else:
        a3 = tile_to(o["speech"], 3.0)
    for N in GAPS:
        gap_bg = tile_to(m["bg_inact"], N) if m["bg_inact"] is not None else None
        gap_q = tile_to(m["quiet"], N)
        conds.append((f"decaybg{N:g}", None if gap_bg is None else np.concatenate([a3, gap_bg])))
        conds.append((f"decayq{N:g}", np.concatenate([a3, gap_q])))
        # CONTROLS: the same gap with NO anchor in front. A long quiet/silent prefix is itself a
        # suppress bias (see silence3), so the anchor's surviving effect is
        # Delta(decay{N}) - Delta(gap{N}), not Delta(decay{N}).
        conds.append((f"gapbg{N:g}", gap_bg))
        conds.append((f"gapq{N:g}", gap_q))
    return conds


# ---------------------------------------------------------------- model
def load_model(config_path: str, ckpt: str, device: torch.device):
    from puresound.config import load_recipe
    from puresound.recipes import init_siso_model
    model = init_siso_model(load_recipe(config_path, expected_task="voice_isolation").model)
    state = torch.load(ckpt, map_location="cpu")
    state = state.get("state_dict", state)
    miss, unexp = model.load_state_dict(state, strict=False)
    print(f"[load] missing={len(miss)} unexpected={len(unexp)} ckpt={ckpt}", flush=True)
    return model.to(device).eval()


@torch.no_grad()
def infer(model, x: np.ndarray, device) -> np.ndarray:
    t = torch.from_numpy(x.astype(np.float32)).view(1, -1).to(device)
    return model(t).detach().cpu().view(-1).numpy()


def cmd_run(args) -> None:
    device = torch.device(args.device)
    items = load_set(Path(args.set_dir), args.limit)
    mat = [build_material(r) for r in items]
    part = partner_index(items)
    same = same_spk_index(items)
    n_bg_fallback = sum(1 for m in mat if m["bg_inact"] is None)
    print(f"{len(items)} utts; bg_inact unavailable for {n_bg_fallback}", flush=True)
    model = load_model(args.config_path, args.ckpt, device)

    out_path = Path(args.out) / f"rows_{args.arm}_{args.tag}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(out_path, "w", encoding="utf-8")
    for i, rec in enumerate(items):
        m = mat[i]
        nf, act = m["nf"], m["act"]
        mix_c = rec["mix"][: nf * FRAME]
        ref_c = rec["ref"][: nf * FRAME]
        if args.arm in ("utt", "ident"):
            sig, target, smask = mix_c, ref_c, act
        else:                                    # lone interferer + noise, user absent
            sig = m["bg_sub"]
            e = frame_energy(sig, nf)
            smask = ~((e < e.max() * 10.0 ** (-REL_DB / 10.0)) | (e < 10.0 ** (ABS_DBFS / 10.0)))
            target = ref_c                       # only used for sisdr bookkeeping (ignored below)
        sisdr_mix = si_sdr(mix_c, ref_c)
        onm = onset_masks(smask)
        conds = (conditions_ident(i, mat, part, same) if args.arm == "ident"
                 else conditions(i, mat, part, args.arm))
        for cond, pad in conds:
            if pad is None and cond != "none":
                fh.write(json.dumps({"tag": args.tag, "arm": args.arm, "id": rec["id"],
                                     "condition": cond, "skipped": True}) + "\n")
                continue
            x = sig if pad is None else np.concatenate([pad, sig])
            off = 0 if pad is None else len(pad)
            y = infer(model, x, device)[off:]
            y = y[: nf * FRAME]
            if len(y) < nf * FRAME:
                y = np.pad(y, (0, nf * FRAME - len(y)))
            row = {"tag": args.tag, "arm": args.arm, "id": rec["id"], "condition": cond,
                   "pad_len_s": round(off / SR, 3),
                   "del_db": db_ratio(y, sig, smask),
                   "leak_db": db_ratio(y, sig, ~smask),
                   "rt60": rec["rt60"], "near_dist": rec["near_dist"],
                   "n_interferers": rec["n_interferers"],
                   "act_sec": round(float(smask.sum()) * FRAME / SR, 2),
                   "inact_sec": round(float((~smask).sum()) * FRAME / SR, 2),
                   "partner": items[part[i]]["id"]}
            for k, mk in onm.items():
                row[f"db_{k}"] = db_ratio(y, sig, mk)
            if args.arm in ("utt", "ident"):
                row["delref_db"] = db_ratio(y, ref_c, act)
                row["sisdr"] = si_sdr(y, target)
                row["sisdri"] = row["sisdr"] - sisdr_mix
            else:
                row["sisdr"] = None
                row["sisdri"] = None
            fh.write(json.dumps(row) + "\n")
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(items)}", flush=True)
            fh.flush()
    fh.close()
    print(f"wrote {out_path}")


# ---------------------------------------------------------------- report
_ORDER = (["none"] + [f"{k}{L:g}" for L in (1.0, 3.0)
           for k in ("self", "selflate", "samespk", "diffspk")] + ["silence3", "bg3", "bgsub3", "quiet3", "self3"]
          + [f"other{L:g}" for L in SWEEP] + [f"self{L:g}" for L in SWEEP]
          + [f"gapq{N:g}" for N in GAPS] + [f"decayq{N:g}" for N in GAPS]
          + [f"gapbg{N:g}" for N in GAPS] + [f"decaybg{N:g}" for N in GAPS])
ORDER = list(dict.fromkeys(_ORDER))


def wilcoxon(d: np.ndarray):
    from scipy.stats import wilcoxon as w
    d = d[np.isfinite(d)]
    d = d[d != 0]
    if len(d) < 6:
        return float("nan"), 0, 0
    st = w(d)
    return float(st.pvalue), int((d < 0).sum()), int((d > 0).sum())


def cmd_report(args) -> None:
    import collections
    tables = []
    for p in args.rows:
        rows = [json.loads(l) for l in open(p, encoding="utf-8")]
        tag, arm = rows[0]["tag"], rows[0].get("arm", "utt")
        by = collections.defaultdict(dict)   # id -> cond -> row
        for r in rows:
            if r.get("skipped"):
                continue
            by[r["id"]][r["condition"]] = r
        base = {i: c.get("none") for i, c in by.items()}
        main_name = "supp_db" if arm == "bgonly" else "del_db"
        thr = -6.0 if arm == "bgonly" else -3.0
        hit_name = f"{'pass' if arm == 'bgonly' else 'del'}{'<=' if arm == 'bgonly' else '<'}{thr:g}dB"
        hdr = (f"| condition | n | {main_name} med | Δ vs none | p | ↓/↑ | {hit_name} "
               f"| Δon1 (first 1 s) | p | Δon2 | Δlate | leak/gap med | Δleak | sisdri Δ |")
        lines = [f"\n## {tag} / arm={arm}   (n utts = {len(by)})", "", hdr,
                 "|" + "---|" * 14]
        f = (lambda v: f"{np.median(v):+.2f}" if len(v) else "--")
        for cond in ORDER:
            ds, dds, ls, dls, dss = [], [], [], [], []
            don = {"on1": [], "on2": [], "late": []}
            hits = 0
            for i, c in by.items():
                r = c.get(cond); b = base[i]
                if r is None or b is None:
                    continue
                ds.append(r["del_db"]); dds.append(r["del_db"] - b["del_db"])
                hits += int(r["del_db"] < thr)
                if r["leak_db"] is not None and b["leak_db"] is not None:
                    ls.append(r["leak_db"]); dls.append(r["leak_db"] - b["leak_db"])
                for k in don:
                    if r.get(f"db_{k}") is not None and b.get(f"db_{k}") is not None:
                        don[k].append(r[f"db_{k}"] - b[f"db_{k}"])
                if r.get("sisdri") is not None and b.get("sisdri") is not None:
                    dss.append(r["sisdri"] - b["sisdri"])
            if not ds:
                continue
            pd_, dn, up = wilcoxon(np.array(dds))
            p1_, _, _ = wilcoxon(np.array(don["on1"])) if don["on1"] else (float("nan"), 0, 0)
            lines.append(
                f"| {cond} | {len(ds)} | {np.median(ds):+.2f} | {np.median(dds):+.2f} | "
                f"{pd_:.2g} | {dn}/{up} | {hits} | {f(don['on1'])} | {p1_:.2g} | "
                f"{f(don['on2'])} | {f(don['late'])} | {f(ls)} | {f(dls)} | {f(dss)} |")
        # anchor effect corrected for the gap-only control
        lines += ["", "### decay, corrected for the gap-only control "
                  "(Δdecay − Δgap, per utterance; the anchor's surviving effect)", "",
                  "| gap | N | n | Δdecay | Δgap | corrected | p (corrected) | ↓/↑ |",
                  "|---|---|---|---|---|---|---|---|"]
        for kind in ("q", "bg"):
            for N in GAPS:
                dec, gap, corr = [], [], []
                for i, c in by.items():
                    rd = c.get(f"decay{kind}{N:g}"); rg = c.get(f"gap{kind}{N:g}"); b = base[i]
                    if rd is None or rg is None or b is None:
                        continue
                    dec.append(rd["del_db"] - b["del_db"])
                    gap.append(rg["del_db"] - b["del_db"])
                    corr.append(rd["del_db"] - rg["del_db"])
                if not corr:
                    continue
                pc, dn, up = wilcoxon(np.array(corr))
                lines.append(f"| {kind} | {N:g} | {len(corr)} | {np.median(dec):+.2f} | "
                             f"{np.median(gap):+.2f} | **{np.median(corr):+.2f}** | "
                             f"{pc:.2g} | {dn}/{up} |")
        tables.append("\n".join(lines))
    print("\n".join(tables))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("config_path"); r.add_argument("--ckpt", required=True)
    r.add_argument("--tag", required=True); r.add_argument("--out", required=True)
    r.add_argument("--set-dir", default=str(RECIPE_DIR / "data_report/wer_set_moderate_test"))
    r.add_argument("--device", default="cuda:0"); r.add_argument("--limit", type=int, default=None)
    r.add_argument("--arm", default="utt", choices=["utt", "bgonly", "ident"])
    r.set_defaults(fn=cmd_run)
    q = sub.add_parser("report")
    q.add_argument("rows", nargs="+")
    q.set_defaults(fn=cmd_report)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
