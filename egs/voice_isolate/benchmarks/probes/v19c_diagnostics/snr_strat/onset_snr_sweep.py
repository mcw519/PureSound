"""Onset deletion vs CONTROLLED local SNR on synthetic data: mix' = ref + a*(mix-ref), background rescaled so the
utterance-level SNR (ref vs background over ref-active frames) hits each target. Same models, both WER sets."""
import json, sys, os, numpy as np, torch, soundfile as sf
os.chdir("/home/milowu/A4Audio/PureSound/egs/voice_isolate"); sys.path.insert(0, "/home/milowu/A4Audio/PureSound")
from puresound.config import load_recipe
from puresound.recipes import init_siso_model
SR=16000; W=0.5; HOP=160; TARGETS=[-15,-10,-5,0,5,10]
CK={"v8":("config/infer_dpcrn.yaml","pretrained_ckpt/dpcrn_v8.ckpt"),
    "v16":("config/infer_dpcrn.yaml","/work/any_exp_link/puresound_exp/dpcrn_v16_lengthmix/lightning_logs/version_0/checkpoints/epoch=19-step=10000.ckpt")}
def active(ref):
    n=len(ref)//HOP; e=(ref[:n*HOP].reshape(n,HOP)**2).mean(1); a=e>e.max()*10**(-40/10); a=np.repeat(a,HOP); return np.concatenate([a,np.zeros(len(ref)-len(a),bool)])
def g_db(enh,ref,m):
    e,r=enh[m].astype(np.float64),ref[m].astype(np.float64); rr=(r*r).sum(); p=(e*r).sum()/max(rr,1e-12)
    return 10*np.log10(max(p,1e-3)**2)   # floor -60 dB; negative projection = fully deleted
def onset_masks(ref):
    a=active(ref); idx=np.flatnonzero(a); t0=idx[0]; on=np.zeros(len(a),bool); on[t0:t0+int(W*SR)]=True; on&=a; return a,on,a&~on
rows=[]
for tag,(cfg,ck) in CK.items():
    model=init_siso_model(load_recipe(cfg,expected_task="voice_isolation").model); st=torch.load(ck,map_location="cpu")
    model.reload_checkpoint(st.get("state_dict",st),load_loss_func=False); model=model.cuda(0).eval()
    for sname in ("wer_set_moderate_test","indomain_wer_set"):
        items=[json.loads(l) for l in open(f"data_report/{sname}/manifest.jsonl")][:120]
        for it in items:
            mix,_=sf.read(f"data_report/{sname}/{it['id']}_mix.wav",dtype="float32"); ref,_=sf.read(f"data_report/{sname}/{it['id']}_ref.wav",dtype="float32")
            n=min(len(mix),len(ref)); mix,ref=mix[:n],ref[:n]; bg=mix-ref
            a,on,rest=onset_masks(ref)
            if on.sum()<int(0.3*W*SR) or rest.sum()<SR: continue
            snr0=10*np.log10(((ref[a]**2).sum()+1e-12)/((bg[a]**2).sum()+1e-12))
            for T in TARGETS:
                alpha=10**((snr0-T)/20); x=ref+alpha*bg; pk=np.abs(x).max()
                if pk>0.99: x=x*(0.99/pk); r2=ref*(0.99/pk)
                else: r2=ref
                with torch.no_grad(): enh=model(torch.from_numpy(x.astype(np.float32)).view(1,-1).cuda(0)).cpu().view(-1).numpy()
                m=min(len(enh),n); rows.append(dict(tag=tag,set=sname,item=it["id"],target_snr=T,snr0=float(snr0),
                    g_on=g_db(enh[:m],r2[:m],on[:m]),g_rest=g_db(enh[:m],r2[:m],rest[:m]),
                    d_on=10*np.log10((enh[:m][on[:m]]**2).mean()/((x[:m][on[:m]]**2).mean()+1e-12)+1e-12),
                    d_rest=10*np.log10((enh[:m][rest[:m]]**2).mean()/((x[:m][rest[:m]]**2).mean()+1e-12)+1e-12)))
    print(tag,"done",flush=True)
json.dump([{k:(float(v) if isinstance(v,(np.floating,np.integer)) else v) for k,v in r.items()} for r in rows],open("/tmp/claude-1001/-home-milowu-A4Audio-PureSound/c52f3b49-8e8f-4f9f-a5ca-862771bd494c/scratchpad/snr_strat/sweep_rows.json","w"))
print(f"\n{'':6s}{'target SNR':>10s} | fg-projection excess g_on-g_rest: median / p10 (n) | g_on median | span-energy excess d_on-d_rest median")
for tag in CK:
    for sname in ("wer_set_moderate_test","indomain_wer_set"):
        for T in TARGETS:
            R=[r for r in rows if r["tag"]==tag and r["set"]==sname and r["target_snr"]==T]
            ex=np.array([r["g_on"]-r["g_rest"] for r in R]); gon=np.array([r["g_on"] for r in R]); de=np.array([r["d_on"]-r["d_rest"] for r in R])
            print(f"{tag:4s} {sname[:8]:8s} {T:+4d} dB | {np.median(ex):+6.2f} / {np.percentile(ex,10):+6.1f} (n={len(R)}) | {np.median(gon):+6.2f} | {np.median(de):+6.2f}")
