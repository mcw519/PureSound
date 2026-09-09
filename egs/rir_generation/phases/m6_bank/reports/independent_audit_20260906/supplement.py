import ast, json, hashlib, random
from pathlib import Path
from typing import Optional
report = Path('/tmp/m6_independent_audit_20260906.json')
j = json.loads(report.read_text())
base = Path('/work/any_exp_link/puresound_exp')
source = ast.parse(Path('puresound/audio/rir/bank/loader.py').read_text())
cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'PreGeneratedRoomBank')
fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_pick')
fn.decorator_list = []
namespace = {'random': random, 'Optional': Optional}
exec(compile(ast.Module(body=[fn], type_ignores=[]), '<repository _pick>', 'exec'), namespace)
for name, folder in [('m6_main','path-events-m4_release'),('m6_boundary','boundary_release')]:
 b = j['banks'][name]
 release = base/'hybrid_rir_16k_m6_20260804'/folder
 r = json.loads((release/'rir_bank_release.json').read_text())
 checks = {}
 for v in r['variants']:
  for key in ['manifest','qc_summary','distribution']:
   p = release/v[key]['path'];checks[v['variant_id']+'/'+key] = hashlib.sha256(p.read_bytes()).hexdigest() == v[key]['file_sha256']
 for rec in r['recipes']:
  for split, desc in (rec.get('split_indexes') or {}).items():
   p = release/desc['path'];checks[rec['recipe_id']+'/'+split] = hashlib.sha256(p.read_bytes()).hexdigest() == desc['sha256']
 b['release_descriptor_hash_checks'] = checks
 b['release'] = r
 fail = []
 for row in b['rooms']:
  m = json.loads(Path(row['file']).read_text())
  pool = [{'channel':c['channel'],'distance':c['distance_m']} for c in m['scene']['channel_map'] if c['label'].startswith('far')]
  chosen = namespace['_pick'](pool, [1.5,4.0], set())
  if not 1.5 <= chosen['distance'] <= 4.0:fail.append({'file':row['file'],'selected_distance':chosen['distance']})
 b['v20_requested_far_1p5_to_4m_out_of_range_scenes'] = len(fail)
 b['v20_out_of_range_examples'] = fail[:3]
 b['manifest']['measured_calibration_evidence_present'] = any(p.get('calibration_report_sha256') for p in b['manifest']['renderer_profiles'])
j['historical_hparams'] = {n: (base/n/'lightning_logs/version_0/hparams.yaml').read_text() for n in ['dpcrn_m6bank_scratch_20260804','dpcrn_m6bank_v8recipe','dpcrn_m6bank_union','dpcrn_m6bank_real']}
j['inspected_source_sha256'] = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ['puresound/audio/impulse_response.py','puresound/audio/rir/bank/loader.py','puresound/audio/rir/scene/sampling.py','puresound/audio/rir/metrics/temporal.py','egs/voice_isolate/config/exp/train_dpcrn_v20_r1a.yaml','egs/voice_isolate/config/exp/train_dpcrn_m6bank_scratch.yaml','egs/voice_isolate/config/exp/train_dpcrn_m6bank_v8recipe.yaml']}
report.write_text(json.dumps(j, indent=2))
print({n:all(b['release_descriptor_hash_checks'].values()) for n,b in j['banks'].items() if 'release_descriptor_hash_checks' in b})
