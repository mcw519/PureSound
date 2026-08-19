"""Does releasing the blend on short in-session near spans actually damage anything?

The b trajectory sinks below the dead zone on 31.2% of frames in near spans under
2.5 s, so the mechanism would raise dry_blend toward 1.0 while the user is talking.
That is only harmful if the MASK is also wrong there -- the blend is insurance, and
dropping insurance costs nothing on a claim that was never going to be made.

Three systems, one forward pass. The mask output is identical in all of them; only
the blend differs:

  fixed     0.9 * enh + 0.1 * mix     what ships today, a flat -20 dB floor
  b-driven  d(t) * enh + (1-d(t)) * mix   the proposal, d from b
  pure      enh                       no insurance at all -- the WORST CASE, and
                                      the decisive one: b-driven is bounded between
                                      fixed and pure, so if pure does no damage the
                                      exposure is nil whatever d(t) does.

Scored with eval_realcase's own definitions: preservation_db = out - in over the
keep spans, KEEP-VIOLATION below -3 dB.
"""
import json, math, os, pathlib, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/milowu/A4Audio/PureSound")
sys.path.insert(0, str(pathlib.Path(sys.argv[0]).parent))
import numpy as np, torch
from puresound.audio.io import AudioIO
from puresound.config import load_recipe
from puresound.recipes import init_siso_model
from b_traj import integrate

S = pathlib.Path(os.environ.get("PROBE_WORK", pathlib.Path(sys.argv[0]).parent))
CASES = pathlib.Path("data_report/field_cases/test_vector_cases")
TAU_UP, TAU_DN, B_LO, B_HI = 0.05, 1.0, 0.35, 0.75
BASE, TOP = 0.9, 0.995
KEEP_VIOLATION_DB = -3.0


def span_dbfs(wav, spans, sr, limit):
    tot = n = 0.0
    for a, b in spans:
        i, j = int(a * sr), min(int(b * sr), limit)
        if j > i:
            tot += float(wav[..., i:j].square().sum()); n += j - i
    return 10.0 * math.log10(tot / n + 1e-12) if n > 0 else float("nan")


if __name__ == "__main__":
    dev = torch.device("cuda:0")
    model = init_siso_model(load_recipe("config/infer_dpcrn.yaml",
                                        expected_task="voice_isolation").model)
    ck = torch.load("pretrained_ckpt/dpcrn_v8.ckpt", map_location="cpu")
    model.reload_checkpoint(ck["state_dict"], load_loss_func=False)
    model = model.to(dev).eval()

    windows = json.loads((CASES / "windows.json").read_text())
    probs = np.load(S / "probs.npz")
    feats = np.load(S / "all_v8.npz")

    print("Per-span keep preservation, three blends. 'pure' is the worst case.\n")
    hdr = f"{'span':>16s} {'len':>6s} {'median b':>9s} | " \
          f"{'fixed':>8s} {'b-driven':>9s} {'pure':>8s} | {'delta':>7s}"
    for clip in ("90d_session", "270d_session"):
        spec = windows[clip]
        mix, sr = AudioIO.open(f_path=str(CASES / f"{clip}_raw.wav"),
                               target_lvl=None, resample_to=16000)
        mix = mix.view(1, -1)
        with torch.no_grad():
            enh = model(mix.to(dev), dry_blend=1.0).detach().cpu().view(1, -1)
        n = min(enh.shape[-1], mix.shape[-1])
        enh, mixn = enh[..., :n], mix[..., :n]

        fps = float(feats[f"{clip}_fps"][0])
        b = integrate(probs[clip], fps, TAU_UP, TAU_DN)
        d = BASE + np.clip((B_HI - b) / (B_HI - B_LO), 0, 1) * (TOP - BASE)
        hop = int(round(sr / fps))
        dsamp = torch.from_numpy(np.repeat(d, hop)).float()
        dsamp = (torch.cat([dsamp, dsamp[-1:].expand(max(0, n - len(dsamp)))])
                 if len(dsamp) < n else dsamp[:n]).view(1, -1)

        sysd = {
            "fixed": (BASE * enh + (1 - BASE) * mixn).clamp(-1, 1),
            "b-driven": (dsamp * enh + (1 - dsamp) * mixn).clamp(-1, 1),
            "pure": enh.clamp(-1, 1),
        }

        print(f"\n=== {clip} ===  KEEP spans (user is talking)")
        print(hdr)
        for i, (x, y) in enumerate(spec.get("keep", [])):
            sp = [(x, y)]
            lo, hi = max(0, int(x*fps)), min(len(b), int(y*fps))
            if hi - lo < 5:
                continue
            din = span_dbfs(mixn, sp, sr, n)
            v = {k: span_dbfs(w, sp, sr, n) - din for k, w in sysd.items()}
            flag = "  <-- KEEP-VIOLATION" if v["pure"] < KEEP_VIOLATION_DB else ""
            print(f"{clip[:4]+' near '+str(i):>16s} {y-x:5.1f}s "
                  f"{np.median(b[lo:hi]):9.3f} | {v['fixed']:8.2f} {v['b-driven']:9.2f} "
                  f"{v['pure']:8.2f} | {v['pure']-v['fixed']:+7.2f}{flag}")

        print(f"\n--- {clip} SUPPRESS spans (bystander) ---")
        print(hdr.replace("keep", "supp"))
        for i, (x, y) in enumerate(spec.get("suppress", [])):
            sp = [(x, y)]
            lo, hi = max(0, int(x*fps)), min(len(b), int(y*fps))
            if hi - lo < 5:
                continue
            din = span_dbfs(mixn, sp, sr, n)
            v = {k: span_dbfs(w, sp, sr, n) - din for k, w in sysd.items()}
            print(f"{clip[:4]+' far  '+str(i):>16s} {y-x:5.1f}s "
                  f"{np.median(b[lo:hi]):9.3f} | {v['fixed']:8.2f} {v['b-driven']:9.2f} "
                  f"{v['pure']:8.2f} | {v['b-driven']-v['fixed']:+7.2f}")

        print(f"\n  whole-clip keep : " + "  ".join(
            f"{k} {span_dbfs(w, spec.get('keep', []), sr, n) - span_dbfs(mixn, spec.get('keep', []), sr, n):+.2f}"
            for k, w in sysd.items()))
        print(f"  whole-clip supp : " + "  ".join(
            f"{k} {span_dbfs(w, spec.get('suppress', []), sr, n) - span_dbfs(mixn, spec.get('suppress', []), sr, n):+.2f}"
            for k, w in sysd.items()))
