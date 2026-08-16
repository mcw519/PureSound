"""Score a checkpoint on the frozen BUT real-RIR WER set (real LibriTTS transcripts).

WER ground truth = LibriTTS transcript (reliable, NOT whisper). whisper is only the
recognizer applied to mix and enhanced audio. Reports wer_mix vs wer_enh: does the
model recover foreground intelligibility on REAL room acoustics + far interferers?
mix transcripts are cached (model-independent) so re-runs only transcribe enhanced.

Usage (ASR backend is switchable via --asr):
    # local whisper (default): faster-whisper > openai-whisper
    uv run python scripts/eval_wer.py config/exp/train_dpcrn_curriculum_expand.yaml \
        --ckpt <ckpt> --set-dir data_report/but_wer_set --device cuda --asr-model small
    # stronger local recognizer (lowers the reverb floor; GPU recommended)
    ... --asr faster-whisper --asr-model large-v3
    # Azure cloud STT (needs SPEECH_KEY + SPEECH_REGION env vars)
    SPEECH_KEY=... SPEECH_REGION=eastus ... --asr azure
Mix transcripts + raw transcripts are cached/saved per backend (mix_hyp_<backend>.jsonl),
so switching --asr never reuses another backend's cache. Numbers are only comparable
within the same backend (compare enh-vs-mix, not across recognizers).
"""
from __future__ import annotations
import argparse, os, sys, json, statistics as st
from pathlib import Path
RECIPE_DIR = Path(__file__).resolve().parents[1]; REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path: sys.path.insert(0, str(REPO))
import numpy as np, torch, soundfile as sf  # noqa: E402
from puresound.recipes import init_siso_model, load_siso_recipe_config  # noqa: E402

def si_sdr(est, ref, eps=1e-8):
    est = est.reshape(-1)-est.reshape(-1).mean(); ref = ref.reshape(-1)-ref.reshape(-1).mean()
    a=(est@ref)/((ref@ref)+eps); s=a*ref; e=est-s
    return float(10.0*torch.log10(((s@s)+eps)/((e@e)+eps)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config_path"); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--set-dir", default="data_report/but_wer_set")
    ap.add_argument("--device", default="cpu"); ap.add_argument("--asr-model", default="small")
    ap.add_argument("--asr", default="auto",
                    choices=["auto", "faster-whisper", "openai-whisper", "azure"],
                    help="ASR backend. auto=faster-whisper>openai-whisper (local). "
                         "azure needs SPEECH_KEY+SPEECH_REGION env. --asr-model is the whisper "
                         "size (e.g. small / large-v3); ignored by azure.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-blend", type=float, default=1.0,
                    help="over-suppression relief: out=a*enh+(1-a)*mix; 1.0=off (default)")
    ap.add_argument("--spec-floor", type=float, default=0.0,
                    help="spectral floor: |enh|>=floor*|mix| per bin (keeps enh phase); 0.0=off (default)")
    args = ap.parse_args()
    cfg_path=str(Path(args.config_path).resolve()); ckpt=str(Path(args.ckpt).resolve())
    sd=Path(args.set_dir).resolve(); os.chdir(RECIPE_DIR); torch.manual_seed(0)
    items=[json.loads(l) for l in open(sd/"manifest.jsonl",encoding="utf-8")]
    if args.limit: items=items[:args.limit]

    model=init_siso_model(load_siso_recipe_config(cfg_path)[5])
    state=torch.load(ckpt,map_location="cpu")["state_dict"]
    miss,unexp=model.load_state_dict(state,strict=False)
    print(f"[load] missing={len(miss)} unexpected={len(unexp)}; ckpt={ckpt}",flush=True)
    model.eval().to(args.device)

    sys.path.insert(0,str(RECIPE_DIR/"scripts"))
    from eval_dawn_chorus import init_asr
    import jiwer
    from whisper.normalizers import EnglishTextNormalizer
    _NORM = EnglishTextNormalizer()   # canonicalises numbers / contractions / abbrev / case / punct
    def norm_wer(refs, hyps):
        rt = jiwer.Compose([jiwer.ReduceToListOfListOfWords()])
        R=[_NORM(x) or " " for x in refs]; H=[_NORM(x) or " " for x in hyps]
        o=jiwer.process_words(R,H,reference_transform=rt,hypothesis_transform=rt)
        n=sum(len(r) for r in o.references)
        return {"wer":o.wer,"substitution_rate":o.substitutions/n,"insertion_rate":o.insertions/n,
                "deletion_rate":o.deletions/n,"n_ref_words":n}
    name,transcribe=init_asr(args.asr,args.asr_model,args.device)
    if transcribe is None: raise SystemExit(f"ASR backend '{args.asr}' unavailable (not installed?)")
    print(f"ASR recognizer: {name} | normalizer: Whisper EnglishTextNormalizer",flush=True)
    asr_slug=name.replace("/","_").replace(":","_")   # cache/outputs keyed by ACTUAL backend, not just model size

    # cache model-independent mix transcripts (per-backend so switching ASR never reuses a stale cache)
    mix_cache=sd/f"mix_hyp_{asr_slug}.jsonl"
    mix_hyp={}
    if mix_cache.exists():
        for l in open(mix_cache,encoding="utf-8"): d=json.loads(l); mix_hyp[d["id"]]=d["hyp"]
        print(f"loaded {len(mix_hyp)} cached mix transcripts")

    refs,hyp_mix,hyp_enh,hyp_ref,sisdri=[],[],[],[],[]
    by_itf={}
    cache_fh = None if mix_cache.exists() else open(mix_cache,"w",encoding="utf-8")
    sr=16000
    with torch.no_grad():
        for i,it in enumerate(items):
            mix,_=sf.read(sd/f"{it['id']}_mix.wav"); ref,_=sf.read(sd/f"{it['id']}_ref.wav")
            mt=torch.tensor(mix,dtype=torch.float32,device=args.device).reshape(1,-1)
            enh=model(mt,dry_blend=args.dry_blend,spec_floor=args.spec_floor).reshape(-1)
            T=min(enh.shape[-1],mt.shape[-1],len(ref))
            rt=torch.tensor(ref,dtype=torch.float32)
            ssi=si_sdr(enh[...,:T].cpu(),rt[...,:T])-si_sdr(mt[...,:T].cpu(),rt[...,:T])
            sisdri.append(ssi)
            refs.append(it["transcript"])
            if it["id"] in mix_hyp: hm=mix_hyp[it["id"]]
            else:
                hm=transcribe(mix.astype(np.float32),sr); mix_hyp[it["id"]]=hm
                if cache_fh: cache_fh.write(json.dumps({"id":it["id"],"hyp":hm},ensure_ascii=False)+"\n"); cache_fh.flush()
            hyp_mix.append(hm)
            hyp_enh.append(transcribe(enh[...,:T].cpu().numpy().astype(np.float32),sr))
            hyp_ref.append(transcribe(np.asarray(ref[:T],dtype=np.float32),sr))   # reverb floor (clean near-reverb fg)
            by_itf.setdefault(it["n_interferers"],[]).append((it["transcript"],hyp_mix[-1],hyp_enh[-1]))
            if (i+1)%25==0: print(f"  {i+1}/{len(items)}",flush=True)
    if cache_fh: cache_fh.close()
    # save raw transcripts so WER can be recomputed offline with any normalizer
    tag = Path(ckpt).stem
    with open(sd/f"transcripts_{asr_slug}_{tag}.jsonl","w",encoding="utf-8") as fh:
        for it,rf,hm,he in zip(items,refs,hyp_mix,hyp_enh):
            fh.write(json.dumps({"id":it["id"],"ref":rf,"mix":hm,"enh":he,"n_interferers":it["n_interferers"]},ensure_ascii=False)+"\n")

    # Per-utterance edit counts, so the enh-vs-mix difference can be given an interval
    # instead of a bare point estimate. A 200-utterance set at WER ~0.55 resolves about
    # +-0.03; quoting a 0.02 "win" from it without the interval is reading noise (this
    # bit the v8/v9/v10 comparison -- see benchmarks/wer_sets/README.md).
    def _counts(refs_, hyps_):
        rt = jiwer.Compose([jiwer.ReduceToListOfListOfWords()])
        out=[]
        for r,h in zip(refs_,hyps_):
            o=jiwer.process_words([_NORM(r) or " "],[_NORM(h) or " "],
                                  reference_transform=rt,hypothesis_transform=rt)
            out.append((o.substitutions+o.insertions+o.deletions, sum(len(x) for x in o.references)))
        return out

    def _ci(rows, n_boot=4000, seed=0):
        """95% bootstrap interval on WER(enh) - WER(mix), resampling utterances."""
        import random
        rng=random.Random(seed); vals=[]
        for _ in range(n_boot):
            s=[rows[rng.randrange(len(rows))] for _ in range(len(rows))]
            n=sum(x[2] for x in s)
            vals.append((sum(x[1] for x in s)-sum(x[0] for x in s))/n if n else float("nan"))
        vals.sort()
        return vals[int(0.025*n_boot)], vals[int(0.975*n_boot)]

    def _delta_line(refs_, hm_, he_, label=""):
        cm=_counts(refs_,hm_); ce=_counts(refs_,he_)
        rows=[(m[0],e[0],m[1]) for m,e in zip(cm,ce)]
        n=sum(r[2] for r in rows)
        d=(sum(r[1] for r in rows)-sum(r[0] for r in rows))/n
        lo,hi=_ci(rows)
        flag="" if (hi<0 or lo>0) else "   <- INSIDE NOISE: this set cannot resolve this difference"
        return f"{label}delta {d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]{flag}"

    wm=norm_wer(refs,hyp_mix); we=norm_wer(refs,hyp_enh); wr=norm_wer(refs,hyp_ref)
    print("="*64); print(f"BUT real-RIR WER benchmark (real LibriTTS transcripts, Whisper-normalized, n={len(refs)})"); print("="*64)
    print(f"SI-SDRi vs near-reverb ref : mean {st.mean(sisdri):+.2f} / median {st.median(sisdri):+.2f} dB  (secondary; ref=full near-reverb, not early)")
    print(f"WER reverb floor (clean near-reverb fg, no itf/noise): {wr['wer']:.3f}  <- benchmark ceiling; if high, reverb-saturated")
    print(f"WER mix      : {wm['wer']:.3f}  (sub {wm['substitution_rate']:.3f} / ins {wm['insertion_rate']:.3f} / del {wm['deletion_rate']:.3f})")
    print(f"WER enhanced : {we['wer']:.3f}  (sub {we['substitution_rate']:.3f} / ins {we['insertion_rate']:.3f} / del {we['deletion_rate']:.3f})")
    print(f"-> enhancement {'REDUCES' if we['wer']<wm['wer'] else 'RAISES'} WER by {abs(we['wer']-wm['wer']):.3f} vs mix")
    print(f"   {_delta_line(refs,hyp_mix,hyp_enh)}")
    print(f"   headroom on this set: mix {wm['wer']:.3f} - reverb floor {wr['wer']:.3f} = {wm['wer']-wr['wer']:.3f};"
          f" this checkpoint captured {100*(wm['wer']-we['wer'])/max(wm['wer']-wr['wer'],1e-9):.0f}% of it")
    print("-- by n_interferers --")
    for k in sorted(by_itf):
        trip=by_itf[k]; r=[t[0] for t in trip]; hm=[t[1] for t in trip]; he=[t[2] for t in trip]
        print(f"   {int(k)} itf (n={len(trip)}): WER mix {norm_wer(r,hm)['wer']:.3f} -> enh {norm_wer(r,he)['wer']:.3f}")
        print(f"        {_delta_line(r,hm,he)}")

if __name__ == "__main__":
    main()
