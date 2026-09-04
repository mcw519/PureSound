"""Onset deletion vs LOCAL SNR: is the real-vs-synthetic gap an SNR artifact?
Per utterance: onset window = first W s of ref-active frames, rest = remaining ref-active frames.
  g = 10log10((<enh,ref>/<ref,ref>)^2)          foreground projection gain (user's own signal retained)
  snr = 10log10(<ref,ref>/<mix-ref,mix-ref>)    local SNR (interferers+noise) over the same samples
  excess = g_on - g_rest
Real: Dawn cache (v8, v16; condition none). Synthetic: moderate + indomain WER sets, models run here."""
import json, sys, os, glob
import numpy as np, torch
os.chdir("/home/milowu/A4Audio/PureSound/egs/voice_isolate"); sys.path.insert(0, "/home/milowu/A4Audio/PureSound")
from puresound.config import load_recipe
from puresound.recipes import init_siso_model
import soundfile as sf
SR=16000; W=0.5; HOP=160
CACHE="/tmp/claude-1001/-home-milowu-A4Audio-PureSound/c52f3b49-8e8f-4f9f-a5ca-862771bd494c/scratchpad/agcache"
OUT="/tmp/claude-1001/-home-milowu-A4Audio-PureSound/c52f3b49-8e8f-4f9f-a5ca-862771bd494c/scratchpad/snr_strat"
CK={"v8":("config/infer_dpcrn.yaml","pretrained_ckpt/dpcrn_v8.ckpt"),
    "v16":("config/infer_dpcrn.yaml","/work/any_exp_link/puresound_exp/dpcrn_v16_lengthmix/lightning_logs/version_0/checkpoints/epoch=19-step=10000.ckpt")}
def active(ref):
    n=len(ref)//HOP; e=(ref[:n*HOP].reshape(n,HOP)**2).mean(1); a=e>e.max()*10**(-40/10)
    return np.repeat(a,HOP)
def g_db(enh,ref,m):
    e,r=enh[m].astype(np.float64),ref[m].astype(np.float64); rr=(r*r).sum()
    return 10*np.log10(max((e*r).sum()/max(rr,1e-12),1e-6)**2) if rr>0 else np.nan
def snr_db(mix,ref,m):
    r=ref[m].astype(np.float64); n=(mix[m]-ref[m]).astype(np.float64)
    return 10*np.log10(((r*r).sum()+1e-12)/((n*n).sum()+1e-12))
def measure(mix,enh,ref,off=0):
    n=min(len(mix),len(enh),len(ref)); mix,enh,ref=mix[off:n],enh[off:n],ref[off:n]
    a=active(ref); n2=len(a); mix,enh,ref=mix[:n2],enh[:n2],ref[:n2]
    idx=np.flatnonzero(a); 
    if len(idx)<int(1.5*W*SR): return None
    t0=idx[0]; on=np.zeros(n2,bool); on[t0:t0+int(W*SR)]=True; on&=a; rest=a&~on
    if rest.sum()<SR*0.5: return None
    return dict(g_on=g_db(enh,ref,on),g_rest=g_db(enh,ref,rest),snr_on=snr_db(mix,ref,on),snr_rest=snr_db(mix,ref,rest),snr_utt=snr_db(mix,ref,a))
rows=[]
for tag in CK:
    for f in sorted(glob.glob(f"{CACHE}/{tag}/dawn/*__none.npz")):
        d=np.load(f); m=json.loads(str(d["meta"])); r=measure(d["mix"].astype(np.float32),d["enh"].astype(np.float32),d["ref"].astype(np.float32))
        if r: rows.append(dict(tag=tag,world="real",set="dawn",item=m["id"],**r))
    cfg,ck=CK[tag]; model=init_siso_model(load_recipe(cfg,expected_task="voice_isolation").model)
    st=torch.load(ck,map_location="cpu"); model.reload_checkpoint(st.get("state_dict",st),load_loss_func=False); model=model.cuda(0).eval()
    for sname in ("wer_set_moderate_test","indomain_wer_set"):
        items=[json.loads(l) for l in open(f"data_report/{sname}/manifest.jsonl")]
        for it in items:
            mix,_=sf.read(f"data_report/{sname}/{it['id']}_mix.wav",dtype="float32"); ref,_=sf.read(f"data_report/{sname}/{it['id']}_ref.wav",dtype="float32")
            with torch.no_grad(): enh=model(torch.from_numpy(mix).view(1,-1).cuda(0)).cpu().view(-1).numpy()
            r=measure(mix,enh,ref)
            if r: rows.append(dict(tag=tag,world="synth",set=sname,item=it["id"],**r))
    print(tag,"done",len(rows),flush=True)
json.dump(rows,open(f"{OUT}/rows.json","w"))
# ---- report
import collections
bins=[-100,-5,0,5,10,15,20,100]; lab=lambda s: next(f"[{bins[i]},{bins[i+1]})" for i in range(len(bins)-1) if bins[i]<=s<bins[i+1])
print("\nonset-window local SNR distribution (median / p10 / p90) and overall excess:")
for tag in CK:
    for world,sname in (("real","dawn"),("synth","wer_set_moderate_test"),("synth","indomain_wer_set")):
        R=[r for r in rows if r["tag"]==tag and r["set"]==sname]; s=np.array([r["snr_on"] for r in R]); ex=np.array([r["g_on"]-r["g_rest"] for r in R])
        print(f"  {tag:4s} {sname:22s} n={len(R):3d} snr_on med {np.median(s):+6.1f} p10 {np.percentile(s,10):+6.1f} p90 {np.percentile(s,90):+6.1f} | snr_rest med {np.median([r['snr_rest'] for r in R]):+6.1f} | excess med {np.median(ex):+6.2f} p10 {np.percentile(ex,10):+6.1f}")
print("\nexcess (g_on - g_rest, dB) by onset-window SNR bin: median (n)")
for tag in CK:
    print(f"  == {tag}")
    hdr=f"  {'bin':12s}"+"".join(f"{s:>28s}" for s in ("dawn(real)","moderate(synth)","indomain(synth)")); print(hdr)
    for i in range(len(bins)-1):
        line=f"  [{bins[i]:>4},{bins[i+1]:>4}) "
        for sname in ("dawn","wer_set_moderate_test","indomain_wer_set"):
            R=[r for r in rows if r["tag"]==tag and r["set"]==sname and bins[i]<=r["snr_on"]<bins[i+1]]
            if len(R)>=5: ex=np.array([r["g_on"]-r["g_rest"] for r in R]); line+=f"{np.median(ex):+8.2f} p10 {np.percentile(ex,10):+7.1f} (n={len(R):3d})"
            else: line+=f"{'-':>10s} (n={len(R):3d})".rjust(28)
        print(line)
    # also: does excess track the DROP in SNR from onset to rest?
    for sname in ("dawn","wer_set_moderate_test","indomain_wer_set"):
        R=[r for r in rows if r["tag"]==tag and r["set"]==sname]
        x=np.array([r["snr_on"]-r["snr_rest"] for r in R]); y=np.array([r["g_on"]-r["g_rest"] for r in R])
        from scipy.stats import spearmanr; rho,p=spearmanr(x,y)
        print(f"  {sname:22s} snr_on-snr_rest med {np.median(x):+6.2f} | spearman(excess, snr_on-snr_rest) rho {rho:+.2f} p {p:.1e} | spearman(excess, snr_on) rho {spearmanr([r['snr_on'] for r in R],y)[0]:+.2f}")
