"""Paired near/far ordering on close+far synchronized corpora (new chain families).

The corpora (DiPCo / AMI / NOTSOFAR) record the SAME moment on a worn close
mic and a distant array. That yields the one test that needs no threshold and
no diarization model: for a window where speaker k talks alone, does the
close-mic version read MORE near-present than the array version of the same
moment? Within-pair, within-chain -- exactly the comparison the
self-calibration round showed survives when absolute calibration dies.

Reported per checkpoint and corpus:
  * pair ordering accuracy (close > array on the near-head logit)
  * absolute medians per side -- where a fixed threshold 0 would land on this
    never-seen chain family (the calibration offset the gate would face)
  * DistHead fg estimates per side (does the distance slot separate too)

Single-speaker windows come from the corpus's own utterance annotations
(window = one utterance 3.5-10 s with no overlapping utterance from any other
speaker, using each device's own clock).

Usage: corpus_pair_probe.py --corpus dipco --ckpt NAME=PATH [--ckpt ...]
       [--max-per-session 15]   (run from egs/voice_isolate/)
"""
import argparse, glob, json, pathlib, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/milowu/A4Audio/PureSound")
import numpy as np, soundfile as sf, torch

SR = 16000
CORPORA = pathlib.Path("exp/real_e2e_corpora")


def hms(s):
    h, m, sec = s.split(":")
    return int(h) * 3600 + int(m) * 60 + float(sec)


def dipco_pairs(max_per_session):
    """(tag, close_path, close_t0, close_t1, far_path, far_t0, far_t1)"""
    root = CORPORA / "dipco/Dipco"
    out = []
    for tjson in sorted(glob.glob(str(root / "transcriptions/*/S*.json"))):
        utts = json.load(open(tjson))
        sess = utts[0]["session_id"] if utts else "?"
        split = pathlib.Path(tjson).parent.name
        spans = [(hms(u["start_time"]["close-talk"]), hms(u["end_time"]["close-talk"]), u)
                 for u in utts if "close-talk" in u.get("start_time", {})]
        picked = 0
        for t0, t1, u in spans:
            if not 3.5 <= t1 - t0 <= 10.0:
                continue
            solo = all(o is u or oe <= t0 or os_ >= t1
                       for os_, oe, o in spans if o["speaker_id"] != u["speaker_id"]
                       for _ in [0]) if True else True
            solo = True
            for os_, oe, o in spans:
                if o["speaker_id"] != u["speaker_id"] and os_ < t1 and oe > t0:
                    solo = False
                    break
            if not solo:
                continue
            close = root / f"audio/{split}/{u['session_id']}_{u['speaker_id']}.wav"
            far = root / f"audio/{split}/{u['session_id']}_U01.CH1.wav"
            if not close.exists() or not far.exists() or "U01" not in u["start_time"]:
                continue
            out.append((f"{u['session_id']}", str(close), t0, t1, str(far),
                        hms(u["start_time"]["U01"]), hms(u["end_time"]["U01"])))
            picked += 1
            if picked >= max_per_session:
                break
    return out


def ami_pairs(max_per_session):
    """Energy-dominance solo windows: one headset >= 10 dB above every other
    headset over a 6 s window (annotation-free v0; the NXT word annotations are
    the upgrade path). Channels are sample-synced, so the same window indexes
    the array file directly."""
    import itertools
    root = CORPORA / "ami/amicorpus"
    meetings = ["ES2004b", "ES2009c", "ES2014a", "IS1003b", "IS1008a", "IS1006c",
                "TS3005a", "TS3009b", "TS3012c", "EN2002a", "EN2006a", "IN1008"]
    out = []
    for m in meetings:
        adir = root / m / "audio"
        arr = adir / f"{m}.Array1-01.wav"
        heads = [adir / f"{m}.Headset-{k}.wav" for k in range(4)]
        heads = [h for h in heads if h.exists()]
        if not arr.exists() or len(heads) < 2:
            continue
        sigs = []
        for h in heads:
            x, sr = sf.read(h)
            if sr != SR:
                raise ValueError(f"{h}: {sr}")
            sigs.append(x)
        n = min(len(x) for x in sigs)
        win, step = 6 * SR, 3 * SR
        grid = []
        for a in range(0, n - win, step):
            rms = [20 * np.log10(np.sqrt(np.mean(x[a:a + win] ** 2)) + 1e-10) for x in sigs]
            order = np.argsort(rms)[::-1]
            if rms[order[0]] > -35.0 and rms[order[0]] - rms[order[1]] >= 10.0:
                grid.append((a, order[0], rms[order[0]] - rms[order[1]]))
        grid.sort(key=lambda g: -g[2])
        picked, used = 0, []
        for a, k, _ in grid:
            if any(abs(a - u) < win for u in used):
                continue
            t0, t1 = a / SR, (a + win) / SR
            out.append((m, str(heads[k]), t0, t1, str(arr), t0, t1))
            used.append(a); picked += 1
            if picked >= max_per_session:
                break
    return out


def notsofar_pairs(max_per_session):
    """GT solo utterances; close = the speaker's own CT wav, far = the first
    multichannel device's ch0. GT times are global meeting seconds."""
    root = CORPORA / "notsofar/benchmark-datasets/train_set/240825.1_train/MTG"
    out = []
    for mdir in sorted(root.iterdir())[:12]:
        gt = mdir / "gt_transcription.json"
        if not gt.exists():
            continue
        utts = json.load(open(gt))
        mcs = sorted(d.name for d in mdir.iterdir() if d.name.startswith("mc_"))
        if not mcs:
            continue
        far = mdir / mcs[0] / "ch0.wav"
        spans = [(u["start_time"], u["end_time"], u) for u in utts]
        picked = 0
        for t0, t1, u in sorted(spans, key=lambda x: -(x[1] - x[0])):
            if not 3.5 <= t1 - t0 <= 10.0:
                continue
            if any(o is not u and o["speaker_id"] != u["speaker_id"]
                   and os_ < t1 and oe > t0 for os_, oe, o in spans):
                continue
            close = mdir / u["ct_wav_file_name"]
            if not close.exists() or not far.exists():
                continue
            out.append((mdir.name, str(close), t0, t1, str(far), t0, t1))
            picked += 1
            if picked >= max_per_session:
                break
    return out


def load_model(ckpt):
    from puresound.config import load_recipe
    from puresound.recipes import init_siso_model
    model = init_siso_model(load_recipe("config/exp/train_dpcrn_v11_presence.yaml",
                                        expected_task="voice_isolation",
                                        expected_purpose="train").model)
    model.load_state_dict(torch.load(ckpt, map_location="cpu")["state_dict"], strict=False)
    return model.eval()


def readout(model, path, t0, t1):
    x, sr = sf.read(path, start=int(t0 * SR), stop=int(t1 * SR))
    if sr != SR:
        raise ValueError(f"{path}: {sr} != {SR}")
    x = torch.from_numpy(np.ascontiguousarray(x)).float().view(1, -1)
    with torch.no_grad():
        model(x)
        lg = model.backbone.last_vad_logits[0].float().numpy()
        dist = float(10.0 ** model.backbone.last_dist_preds[0, 1])
    hop = x.shape[-1] / len(lg)
    xm = x[0].numpy()
    dbfs = np.array([20 * np.log10(np.sqrt(np.mean(
        xm[int(i * hop):max(int(i * hop) + 1, int((i + 1) * hop))] ** 2)) + 1e-10)
        for i in range(len(lg))])
    keep = dbfs > -60.0
    return float(np.median(lg[keep])) if keep.any() else float(np.median(lg)), dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=["dipco", "ami", "notsofar"])
    ap.add_argument("--ckpt", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--max-per-session", type=int, default=15)
    args = ap.parse_args()

    pairs = {"dipco": dipco_pairs, "ami": ami_pairs,
             "notsofar": notsofar_pairs}[args.corpus](args.max_per_session)
    print(f"{args.corpus}: {len(pairs)} solo pairs (close mic vs far device, same moment)\n")

    for spec in args.ckpt:
        name, _, path = spec.partition("=")
        model = load_model(path)
        rows = []
        for tag, cp, ct0, ct1, fp, ft0, ft1 in pairs:
            lc, dc = readout(model, cp, ct0, ct1)
            lf, df = readout(model, fp, ft0, ft1)
            rows.append((tag, lc, lf, dc, df))
        lc = np.array([r[1] for r in rows]); lf = np.array([r[2] for r in rows])
        dc = np.array([r[3] for r in rows]); df = np.array([r[4] for r in rows])
        print(f"### {name}")
        print(f"  pair ordering (close reads nearer): {(lc > lf).mean():.3f}  (n={len(rows)})")
        print(f"  near-head logit median: close {np.median(lc):+.2f} / array {np.median(lf):+.2f} "
              f"(threshold-0 calls close present {100*(lc > 0).mean():.0f}%, array present {100*(lf > 0).mean():.0f}%)")
        print(f"  fg_dist estimate median: close {np.median(dc):.2f} m / array {np.median(df):.2f} m "
              f"(pair-ordered {(dc < df).mean():.3f})")
        per = {}
        for tag, a, b, *_ in rows:
            per.setdefault(tag, []).append(a > b)
        worst = min(per.items(), key=lambda kv: np.mean(kv[1]))
        print(f"  worst session: {worst[0]} ordering {np.mean(worst[1]):.2f}\n")


if __name__ == "__main__":
    main()
