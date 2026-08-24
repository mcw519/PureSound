"""Layer-1 self-calibration, simulated offline: does a session-relative
threshold fix what a fixed threshold gets wrong -- without ever fitting labels?

Background (`v11_ep19_VERDICT.md`): the v11 near-presence head transfers to the
rig chain with the ORDER intact but the OFFSET wrong (present +4.36 vs absent
+0.41, both above the factory threshold 0), and dies on the QVF chain where the
order itself is gone (AUC 0.474, a bystander reads +8.05). Self-calibration can
only re-place a boundary; it cannot resurrect a dead ordering -- and if the
ordering is INVERTED it will actively hurt. This sim measures all three cases
honestly, causally (frame t uses only frames <= t), with no labels touched.

The calibrator: gate frames by audibility (running-floor + margin), accumulate
audible near-head logits, split them Otsu-style into two clusters, and act only
when (a) >= BURN_IN_S of audible evidence, (b) both clusters carry real mass,
(c) the clusters separate at d' >= DPRIME_MIN. Threshold sits keep-biased at
mu_low + KEEP_BIAS * gap; while inactive every frame reads "present" (gate
open), so an inactive calibrator IS today's behaviour.

Verdicts wanted:
  * rig sessions / rig cold-start pairs: does balanced accuracy at the
    self-calibrated threshold approach the oracle where the fixed one fails?
  * lone cold-start clips and the 0d sentinel: the calibrator must REFUSE
    (single cluster) -- its safety property.
  * qvf_scenario3_session: expected to activate on an inverted ordering; the
    keep-damage number is the cost of not having an inversion guard.
"""
import argparse, json, pathlib, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/milowu/A4Audio/PureSound")
import numpy as np
import torch

CASES = pathlib.Path("data_report/field_cases/test_vector_cases")
DEVICE = "cpu"   # set by callers that have a GPU free
CKPT = "exp/dpcrn_v11_presence/lightning_logs/version_2/checkpoints/epoch=19-step=10000.ckpt"

BURN_IN_S = 2.0        # measured: ~1 s of audible speech to settle, 2 s is the safe order
FLOOR_MARGIN_DB = 12.0 # audible = above running 10th-percentile frame level + this
DPRIME_MIN = 2.0       # cluster separation required before acting
MIN_MASS_FRAMES = 100  # ~1 s: a cluster smaller than this is an outlier, not a talker
MIN_MASS_FRAC = 0.08
KEEP_BIAS = 0.3        # threshold at mu_low + 0.3*gap: ambiguity reads "present"
RECOMPUTE_EVERY = 25   # frames (~0.25 s): floor + Otsu cadence
NBINS = 64


def extract(clip, model, seg_s=20.0, sr=16000):
    """Per-frame near-head logits + frame dBFS, 20 s segments like the transfer probe."""
    from puresound.audio.io import AudioIO
    wav, _ = AudioIO.open(f_path=str(CASES / f"{clip}_raw.wav"), target_lvl=None,
                          resample_to=sr)
    wav = wav.view(1, -1)
    logits = []
    step = int(seg_s * sr)
    for a in range(0, wav.shape[-1], step):
        seg = wav[..., a:a + step]
        if seg.shape[-1] < sr // 4:
            break
        with torch.no_grad():
            model(seg.to(DEVICE))
        logits.append(model.backbone.last_vad_logits[0].cpu().numpy())
    lg = np.concatenate(logits)
    hop = wav.shape[-1] / len(lg)
    x = wav[0].numpy()
    rms = np.array([
        np.sqrt(np.mean(x[int(i * hop): max(int(i * hop) + 1, int((i + 1) * hop))] ** 2))
        for i in range(len(lg))
    ])
    dbfs = 20.0 * np.log10(rms + 1e-10)
    return lg, dbfs, len(lg) / (wav.shape[-1] / sr)


def otsu_split(values, nbins=NBINS):
    """Threshold maximising between-class variance; returns (thr, lo_stats, hi_stats)."""
    hist, edges = np.histogram(values, bins=nbins)
    mids = 0.5 * (edges[:-1] + edges[1:])
    w = hist.astype(np.float64)
    total = w.sum()
    if total <= 0:
        return None
    cum = np.cumsum(w)
    cum_mu = np.cumsum(w * mids)
    mu_t = cum_mu[-1] / total
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mu_t * cum - cum_mu) ** 2 / (cum * (total - cum))
    between[~np.isfinite(between)] = -1.0
    k = int(np.argmax(between))
    thr = edges[k + 1]
    lo, hi = values[values <= thr], values[values > thr]
    if len(lo) < 2 or len(hi) < 2:
        return None
    return thr, (lo.mean(), lo.std(), len(lo)), (hi.mean(), hi.std(), len(hi))


class Calibrator:
    """Causal. step() per frame -> (active, threshold)."""

    def __init__(self, fps, min_mass_s=None, min_elapsed_s=0.0):
        self.fps = fps
        self.burn_in = int(BURN_IN_S * fps)
        self.min_mass = (int(min_mass_s * fps) if min_mass_s is not None
                         else MIN_MASS_FRAMES)
        self.min_elapsed = int(min_elapsed_s * fps)
        self.levels = []      # every frame's dBFS, for the running floor
        self.audible = []     # audible frames' logits
        self.active = False
        self.threshold = None
        self.activated_at_frame = None
        self._floor = None
        self._i = 0

    def step(self, logit, dbfs):
        self.levels.append(dbfs)
        if self._i % RECOMPUTE_EVERY == 0:
            self._floor = np.percentile(self.levels, 10)
        if dbfs > self._floor + FLOOR_MARGIN_DB:
            self.audible.append(logit)
        if self._i % RECOMPUTE_EVERY == 0 and len(self.audible) >= self.burn_in:
            split = otsu_split(np.asarray(self.audible))
            if split is not None:
                _, (mu0, sd0, n0), (mu1, sd1, n1) = split
                n = n0 + n1
                mass_ok = (min(n0, n1) >= max(self.min_mass, MIN_MASS_FRAC * n)
                           and self._i >= self.min_elapsed)
                dprime = (mu1 - mu0) / np.sqrt(0.5 * (sd0 ** 2 + sd1 ** 2) + 1e-9)
                if mass_ok and dprime >= DPRIME_MIN:
                    if not self.active:
                        self.activated_at_frame = self._i
                    self.active = True
                    self.threshold = mu0 + KEEP_BIAS * (mu1 - mu0)
                # once active, stay active with the latest passing threshold;
                # if criteria lapse we keep the last threshold (no flapping)
        self._i += 1
        return self.active, self.threshold


def frame_labels(spec, n, fps):
    """1 present / 0 absent / -1 unlabeled, from the session's span lists."""
    y = np.full(n, -1, dtype=np.int8)
    for a, b in spec.get("suppress", []):
        y[max(0, int(a * fps)): min(n, int(b * fps))] = 0
    for a, b in spec.get("keep", []):
        y[max(0, int(a * fps)): min(n, int(b * fps))] = 1
    return y


def auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    both = np.concatenate([pos, neg]); order = both.argsort().argsort() + 1
    rp = order[: len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def balanced_acc(y, pred):
    p, a = pred[y == 1], pred[y == 0]
    if len(p) == 0 or len(a) == 0:
        return float("nan")
    return 0.5 * (p.mean() + (1.0 - a.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--cache", default=None, help="npz cache dir for extracted frames")
    ap.add_argument("--min-mass-s", type=float, default=1.0,
                    help="audible seconds each cluster needs before acting")
    ap.add_argument("--min-elapsed-s", type=float, default=0.0,
                    help="no activation before this much of the session has played")
    args = ap.parse_args()

    from puresound.config import load_recipe
    from puresound.recipes import init_siso_model
    model = init_siso_model(load_recipe("config/exp/train_dpcrn_v11_presence.yaml",
                                        expected_task="voice_isolation",
                                        expected_purpose="train").model)
    sd = torch.load(args.ckpt, map_location="cpu")["state_dict"]
    model.load_state_dict(sd, strict=False)
    model = model.eval()

    windows = json.loads((CASES / "windows.json").read_text())
    cache_dir = pathlib.Path(args.cache) if args.cache else None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)

    def frames_for(clip):
        if cache_dir and (cache_dir / f"{clip}.npz").exists():
            z = np.load(cache_dir / f"{clip}.npz")
            return z["lg"], z["dbfs"], float(z["fps"])
        lg, dbfs, fps = extract(clip, model)
        if cache_dir:
            np.savez(cache_dir / f"{clip}.npz", lg=lg, dbfs=dbfs, fps=fps)
        return lg, dbfs, fps

    # ---------------------------------------------------------------- sessions
    print("=== SESSIONS: fixed threshold vs self-calibrated vs oracle ===")
    print(f"{'session':24s} {'AUC':>5s} {'bacc@0':>7s} {'bacc@cal':>8s} {'bacc@orc':>8s} "
          f"{'act@s':>6s} {'thr':>6s} {'keepdmg@0':>9s} {'keepdmg@cal':>11s} {'suppcatch@cal':>13s}")
    for clip in sorted(k for k in windows if k.endswith("_session")):
        spec = windows[clip]
        lg, dbfs, fps = frames_for(clip)
        n = len(lg)
        y = frame_labels(spec, n, fps)

        cal = Calibrator(fps, min_mass_s=args.min_mass_s, min_elapsed_s=args.min_elapsed_s)
        pred_cal = np.ones(n, dtype=np.float32)   # inactive = open = "present"
        thr_trace = np.full(n, np.nan)
        for i in range(n):
            active, thr = cal.step(lg[i], dbfs[i])
            if active:
                pred_cal[i] = 1.0 if lg[i] >= thr else 0.0
                thr_trace[i] = thr

        audible = dbfs > (np.percentile(dbfs, 10) + FLOOR_MARGIN_DB)
        m = (y >= 0) & audible
        ym, lgm = y[m], lg[m]
        pred_fix = (lgm >= 0.0).astype(np.float32)

        cands = np.quantile(lgm, np.linspace(0.01, 0.99, 197))
        orc = max(balanced_acc(ym, (lgm >= t).astype(np.float32)) for t in cands)

        keep_m, supp_m = m & (y == 1), m & (y == 0)
        kd0 = float((lg[keep_m] < 0.0).mean()) if keep_m.any() else float("nan")
        kdc = float((pred_cal[keep_m] == 0).mean()) if keep_m.any() else float("nan")
        sc = float((pred_cal[supp_m] == 0).mean()) if supp_m.any() else float("nan")

        act_s = cal.activated_at_frame / fps if cal.activated_at_frame is not None else float("nan")
        thr_s = f"{cal.threshold:6.2f}" if cal.threshold is not None else "   n/a"
        print(f"{clip:24s} {auc(lgm[ym == 1], lgm[ym == 0]):5.3f} "
              f"{balanced_acc(ym, pred_fix):7.3f} {balanced_acc(ym, pred_cal[m]):8.3f} "
              f"{orc:8.3f} {act_s:6.1f} {thr_s} {kd0:9.3f} {kdc:11.3f} {sc:13.3f}")

    # ------------------------------------------------- cold-start single clips
    print("\n=== COLD-START clips: the calibrator must refuse on one talker ===")
    counts = {"lone near": [0, 0], "lone far": [0, 0], "double-talk": [0, 0]}
    activated_names = []
    for clip, spec in windows.items():
        if clip.startswith("_") or clip.endswith("_session"):
            continue
        role = ("lone far" if "lone far" in spec["role"]
                else "double-talk" if "double-talk" in spec["role"] else "lone near")
        lg, dbfs, fps = frames_for(clip)
        cal = Calibrator(fps, min_mass_s=args.min_mass_s, min_elapsed_s=args.min_elapsed_s)
        for i in range(len(lg)):
            cal.step(lg[i], dbfs[i])
        counts[role][1] += 1
        if cal.active:
            counts[role][0] += 1
            activated_names.append((clip, role, cal.threshold))
    for role, (act, tot) in counts.items():
        print(f"  {role:12s}: activated on {act}/{tot} clips")
    for clip, role, thr in activated_names:
        print(f"    !! {clip} ({role}) activated, thr {thr:.2f}")
    sent = "0d_near1"
    print(f"  sentinel {sent}: {'ACTIVATED (bad)' if any(c == sent for c, _, _ in activated_names) else 'refused (good)'}")


if __name__ == "__main__":
    main()
