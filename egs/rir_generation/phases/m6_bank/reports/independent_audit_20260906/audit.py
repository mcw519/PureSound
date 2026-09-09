from pathlib import Path
from collections import Counter, defaultdict
import json, hashlib, random
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

BASE=Path('/work/any_exp_link/puresound_exp')
OUT=Path('/tmp/m6_independent_audit_20260906.json')
SEED=20260906

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def stats(x):
 x=np.asarray([v for v in x if v is not None and np.isfinite(v)],float)
 return dict(n=len(x),q05=float(np.quantile(x,.05)),median=float(np.median(x)),q95=float(np.quantile(x,.95)),min=float(x.min()),max=float(x.max())) if len(x) else {'n':0}
def decay(x,sr):
 e=np.cumsum(x[::-1]**2)[::-1]
 if e[0]<=0:return None,None
 db=10*np.log10(np.maximum(e/e[0],1e-30));ix=np.flatnonzero((db<=-5)&(db>=-25))
 if len(ix)<32:return None,None
 t=ix/sr;y=db[ix];tc=t-t.mean();yc=y-y.mean();s=float(tc@yc/(tc@tc));r2=float((tc@yc)**2/((tc@tc)*(yc@yc)))
 return (-60/s if s<0 and r2>=.95 else None),r2

def analyze(name, paths, manifest=None):
 rows=[]; errors=Counter(); roomstats=[]; examples=[]; hash_fail=[]; bands=Counter(); rng=random.Random(SEED)
 print('analyzing',name,len(paths),flush=True)
 for idx,(mp,wp,item) in enumerate(paths):
  m=json.loads(mp.read_text());h,sr=sf.read(wp,always_2d=True,dtype='float64');sc=m['scene'];channels=sc.get('channel_map',[])
  if not np.isfinite(h).all():errors['nonfinite']+=1;continue
  if item:
   for key,path in [('metadata',mp),('rir',wp)]:
    if sha(path)!=item['assets'][key]['sha256']:hash_fail.append(str(path))
  coords=sc.get('receivers',[{}])[0].get('pose',{}).get('position_m')
  roomrow=[]
  for ch in channels:
   k=ch['channel'];distance=ch.get('distance_m');
   if distance is None:errors['missing_distance']+=1;continue
   x=h[:,k];absx=abs(x);peak=int(absx.argmax());pos=int(x.argmax());energy=float(x@x)
   if energy<=0:errors['zero_energy']+=1;continue
   actual_d=float(np.linalg.norm(np.asarray(ch['source_pos'])-coords)) if coords is not None else None
   if actual_d is not None and abs(actual_d-distance)>1e-6:errors['distance_geometry_mismatch']+=1
   c=sc.get('environment',{}).get('sound_speed_m_s',343.)
   expected=int(np.floor(distance/c*sr));first=int(np.flatnonzero(absx>absx.max()*1e-7)[0])
   cut=min(len(x),peak+round(.0025*sr));direct=float(x[peak:cut]@x[peak:cut]);tail=float(x[cut:]@x[cut:]);drr=10*np.log10(max(direct,1e-30)/max(tail,1e-30))
   early=x[:min(len(x),pos+round(.05*sr))];ep=int(abs(early).argmax());gain=float(absx.max()/max(abs(early).max(),1e-30))
   if ep!=peak or abs(gain-1)>1e-6:
    errors['full_early_alignment_or_scale_mismatch']+=1
    if len(examples)<3:examples.append(dict(file=str(wp),channel=k,full_peak=peak,early_peak=ep,gain_ratio=gain))
   t20,r2=decay(x,sr);band_t={}
   for f in [500,1000,2000]:
    sos=butter(4,[f/2**.5,f*2**.5],btype='bandpass',fs=sr,output='sos');t,_=decay(sosfilt(sos,x),sr);band_t[str(f)]=t
   r=dict(file=str(wp),room=sc.get('scene_id',str(mp)),channel=k,label=ch.get('label'),distance=distance,drr_peak_2p5ms=drr,t20=t20,t20_r2=r2,band_t20=band_t,rt60_metadata=sc.get('rt60'),peak_lag_from_geometry_ms=(peak-expected)/sr*1000,first_arrival_error_ms=(first-expected)/sr*1000,peak_abs=float(absx.max()),early_gain_ratio=gain)
   rows.append(r);roomrow.append(r)
  roomstats.append(dict(file=str(mp),scene_rt60=sc.get('rt60'),config_rt60_range=m.get('config',{}).get('rt60_range'),crossover_policy=m.get('crossover',{}).get('policy'),source_convention_preserved=m.get('crossover',{}).get('source_convention_preserved')))
  if idx and idx%32==0: print(name,idx,flush=True)
 def group(rr):
  return dict(channels=len(rr),distance=stats([r['distance'] for r in rr]),drr=stats([r['drr_peak_2p5ms'] for r in rr]),t20=stats([r['t20'] for r in rr]),t20_500=stats([r['band_t20']['500'] for r in rr]),t20_1000=stats([r['band_t20']['1000'] for r in rr]),scene_rt60=stats([r['rt60_metadata'] for r in rr]),peak_lag=stats([r['peak_lag_from_geometry_ms'] for r in rr]),distance_1_to_2=sum(1<=r['distance']<2 for r in rr),distance_over5=sum(r['distance']>5 for r in rr))
 byroom=defaultdict(list)
 for r in rows:byroom[r['room']].append(r)
 x=[];y=[]
 for rs in byroom.values():
  a=np.log10([r['distance'] for r in rs]);b=np.asarray([r['drr_peak_2p5ms'] for r in rs]);x.extend(a-a.mean());y.extend(b-b.mean())
 x=np.asarray(x);y=np.asarray(y)
 slope=float(x@y/(x@x)) if x@x>0 else None
 summary=dict(items=len(paths),errors=dict(errors),hash_fail=hash_fail,all=group(rows),near=group([r for r in rows if r['label'].startswith('near')]),far=group([r for r in rows if r['label'].startswith('far')]),within_item_drr_logdistance_slope=slope,full_early_mismatch_examples=examples,crossover_policies=dict(Counter(r['crossover_policy'] for r in roomstats)),source_convention_preserved=dict(Counter(str(r['source_convention_preserved']) for r in roomstats)))
 print('summary',name,json.dumps(summary),flush=True)
 return dict(summary=summary,rooms=roomstats,channels=rows)

result={'seed':SEED,'method':'Independent soundfile/numpy/scipy audit; 128 acoustic spaces per M6 bank, one uniformly sampled train+pass item per selected space. Legacy comparisons sample 128 files per origin; not balanced by physical room. T20: whole-waveform Schroeder EDC -5..-25 dB OLS, R2 >= .95, no noise correction; synthetic descriptive check, measured values exploratory. DRR: operational peak-forward 2.5 ms definition, not certified direct-path energy. No training, full bank loading or GPU.', 'banks':{}}
space_sets={}
for name,folder in [('m6_main','path-events-m4_release'),('m6_boundary','boundary_release')]:
 release=BASE/'hybrid_rir_16k_m6_20260804'/folder;root=release/'variants/calibrated';manifest=json.loads((root/'rir_bank_manifest.json').read_text());items=manifest['items'];byspace=defaultdict(list);splits=defaultdict(set);byroom=defaultdict(set)
 for i in items:
  splits[i['split']].add(i['acoustic_space_id']);byroom[i['room_id']].add(i['split'])
  if i['split']=='train' and i['qc']['status']=='pass':byspace[i['acoustic_space_id']].append(i)
 rng=random.Random(SEED);chosen=[rng.choice(byspace[k]) for k in rng.sample(sorted(byspace),min(128,len(byspace)))];paths=[(root/i['assets']['metadata']['path'],root/i['assets']['rir']['path'],i) for i in chosen]
 data=analyze(name,paths);data['manifest']=dict(file=str(root/'rir_bank_manifest.json'),sha256=sha(root/'rir_bank_manifest.json'),items=len(items),split_items=dict(Counter(i['split'] for i in items)),split_spaces={k:len(v) for k,v in splits.items()},space_split_overlap=len((splits['train']&splits['test'])|(splits['train']&splits['validation'])|(splits['validation']&splits['test'])),room_cross_split=sum(len(s)>1 for s in byroom.values()),generator=manifest['generator'],renderer_profiles=manifest['renderer_profiles']);result['banks'][name]=data;space_sets[name]=splits
 OUT.write_text(json.dumps(result,indent=2))
for name,root,predicate in [('hybrid_sim',BASE/'hybrid_rir_16k_realfar/items',lambda p:not p.name.startswith('real_')),('measured',BASE/'real_rir_16k_train_view/items',lambda p:True)]:
 paths=sorted(p for p in root.glob('*.json') if predicate(p));rng=random.Random(SEED);sample=rng.sample(paths,min(128,len(paths)));result['banks'][name]=analyze(name,[(p,p.with_suffix('.wav'),None) for p in sample]);result['banks'][name]['candidate_files']=len(paths);OUT.write_text(json.dumps(result,indent=2))
result['cross_release_acoustic_space_overlap']={f'{a}/{b}':len(space_sets['m6_main'][a]&space_sets['m6_boundary'][b]) for a in ['train','validation','test'] for b in ['train','validation','test']}
OUT.write_text(json.dumps(result,indent=2));print('written',str(OUT))
