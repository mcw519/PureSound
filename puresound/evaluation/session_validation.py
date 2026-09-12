"""Fixed v20 session evaluation. No optimizer, training step, or GPU required.

All rates include their denominators. Proximity is an uncalibrated raw scalar,
not metres. Onset projection measures target-correlated amplitude and is reported
alongside residual energy; correlated interference can still bias projection.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def assert_disjoint(train_speakers, valid_speakers):
    overlap = set(train_speakers) & set(valid_speakers)
    if overlap:
        raise ValueError(f'train/validation speaker overlap ({len(overlap)}): {sorted(overlap)[:10]}')
    if not valid_speakers:
        raise ValueError('empty validation speaker pool')


def distribution(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    return {'count': int(a.size), 'mean': float(a.mean()) if a.size else None,
            **{f'p{q}': float(np.percentile(a, q)) if a.size else None for q in (5, 50, 95)}}


def _array(value):
    return value.detach().cpu().numpy().reshape(-1) if isinstance(value, torch.Tensor) else np.asarray(value).reshape(-1)


def presence_metrics(logits, target, *, fps=100., delay_frames=0, threshold=.5):
    """Align prediction[t+delay] with truth[t]; misses are right-censored per run.

    Delay is extra inference-path latency only. Offline DPCRN predictions already
    share the label frame origin, so default 0 does not subtract STFT twice.
    """
    if delay_frames < 0 or not 0 < threshold < 1:
        raise ValueError('delay_frames >= 0 and 0 < threshold < 1 required')
    scores, truth = _array(logits)[delay_frames:], _array(target) > .5
    if fps <= 0 or not np.isfinite(scores).all():
        raise ValueError('positive fps and finite presence readouts required')
    n = min(len(scores), len(truth))
    truth, pred = truth[:n], scores[:n] >= np.log(threshold / (1-threshold))
    edges = np.diff(np.r_[False, truth, False].astype(int))
    delays, censored = [], []
    for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        hits = np.flatnonzero(pred[start:end])
        if hits.size:
            delays.append(float(hits[0] / fps))
        else:
            censored.append(float((end-start) / fps))
    pos, neg = int(truth.sum()), int((~truth).sum())
    fn, fp = int((truth & ~pred).sum()), int((~truth & pred).sum())
    return dict(positive_frames=pos, negative_frames=neg, false_negative_frames=fn,
                false_positive_frames=fp, fnr=fn/pos if pos else None, fpr=fp/neg if neg else None,
                onset_delay_seconds=delays, censored_run_seconds=censored,
                detected_runs=len(delays), censored_runs=len(censored), aligned_frames=n)


def proximity_metrics(proximity, labels, paired=None, *, min_distance_gap=.25,
                      min_turn_frames=1, bounded_limit=None, pair_selection="cross_role"):
    if min_distance_gap < 0 or min_turn_frames < 1 or (bounded_limit is not None and bounded_limit <= 0):
        raise ValueError('invalid proximity metric thresholds')
    if pair_selection not in ('cross_role', 'all'):
        raise ValueError('pair_selection must be cross_role or all')
    p, ids = _array(proximity), _array(labels['turn_id'])
    roles, distances = _array(labels['turn_role']), _array(labels['turn_distance'])
    tracks = [labels.get('user_active'), labels.get('bystander_active')]
    sizes = [len(p), len(ids)]
    if paired is not None:
        sizes.append(len(_array(paired)))
    if all(t is not None for t in tracks):
        sizes.extend(len(_array(t)) for t in tracks)
    n = min(sizes)
    p, ids = p[:n], ids[:n]
    if not np.isfinite(p).all() or (paired is not None and not np.isfinite(_array(paired)[:n]).all()):
        raise ValueError('non-finite proximity readout')
    overlap = ((_array(tracks[0])[:n] > .5) & (_array(tracks[1])[:n] > .5)
               if all(t is not None for t in tracks) else np.zeros(n, dtype=bool))
    means, paired_means = {}, {}
    for k, distance in enumerate(distances):
        mask = (ids == k+1) & ~overlap
        if np.isfinite(distance) and distance > 0 and mask.sum() >= min_turn_frames and roles[k] > 0:
            means[k] = float(p[mask].mean())
            if paired is not None:
                paired_means[k] = float(_array(paired)[:n][mask].mean())
    margins, gaps = [], []
    for u in means:
        for b in means:
            if b <= u or (pair_selection == 'cross_role' and roles[u] == roles[b]) or abs(distances[u]-distances[b]) < min_distance_gap:
                continue
            sign = float(np.sign(distances[b]-distances[u]))
            if sign == 0:
                continue
            margin = sign * (means[u]-means[b])
            margins.append(margin)
            if paired is not None:
                gaps.append(abs(margin-sign*(paired_means[u]-paired_means[b])))
    finite = p[np.isfinite(p)]
    return dict(ordering_pairs=len(margins), ordering_correct=sum(m > 0 for m in margins),
                ordering_accuracy=sum(m > 0 for m in margins)/len(margins) if margins else None,
                margins=margins, cross_chain_gap_absolute_error=gaps, consistency_pairs=len(gaps),
                raw_output=distribution(finite),
                saturation_fraction=float((abs(finite) >= .99*bounded_limit).mean()) if bounded_limit and finite.size else None,
                saturation_limit=bounded_limit)


def target_projection(output, target):
    """Signed target gain and residual-to-target energy, not mixture gain."""
    y, s = _array(output).astype(float), _array(target).astype(float)
    n = min(len(y), len(s)); y, s = y[:n], s[:n]
    energy = float(s @ s)
    if energy < 1e-12:
        return None
    gain = float((y @ s) / energy)
    return {'gain': gain, 'gain_db': float(20*np.log10(max(abs(gain), 1e-8))),
            'residual_to_target_db': float(10*np.log10(max(float((y-gain*s) @ (y-gain*s))/energy, 1e-12)))}


def require_checkpoint_heads(model, state, heads=('proximity_head', 'vad_head')):
    """Refuse random initialized evaluation heads or incompatible head shapes."""
    model_state = model.state_dict()
    for name in heads:
        expected = [key for key in model_state if f'.{name}.' in f'.{key}']
        missing = [key for key in expected if key not in state]
        mismatched = [key for key in expected if key in state and state[key].shape != model_state[key].shape]
        if not expected or missing or mismatched:
            raise ValueError(f'checkpoint lacks compatible {name}: missing={missing}, mismatched={mismatched}')


def persist_batch(root, batch, *, row_id, seed, seconds):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    path = root / f'row_{row_id:05d}.pt'
    if path.exists():
        raise FileExistsError(path)
    torch.save(batch, path)
    return {'file': path.name, 'sha256': digest(path), 'seed': seed, 'seconds': seconds,
            'shape': int(batch['session_shape'].reshape(-1)[0])}


def summarize_rows(rows):
    def summarize(items):
        p = [r['presence'] for r in items]; q = [r['proximity'] for r in items]
        totals = {k: sum(r[k] for r in p) for k in ('positive_frames','negative_frames','false_negative_frames','false_positive_frames','detected_runs','censored_runs')}
        totals['fnr'] = totals['false_negative_frames']/totals['positive_frames'] if totals['positive_frames'] else None
        totals['fpr'] = totals['false_positive_frames']/totals['negative_frames'] if totals['negative_frames'] else None
        totals['onset_delay_seconds'] = distribution([v for r in p for v in r['onset_delay_seconds']])
        totals['censored_run_seconds'] = distribution([v for r in p for v in r['censored_run_seconds']])
        pairs, correct = sum(r['ordering_pairs'] for r in q), sum(r['ordering_correct'] for r in q)
        return dict(rows=len(items), presence=totals, proximity=dict(ordering_pairs=pairs, ordering_correct=correct,
            ordering_accuracy=correct/pairs if pairs else None, margins=distribution([v for r in q for v in r['margins']]),
            cross_chain_gap_absolute_error=distribution([v for r in q for v in r['cross_chain_gap_absolute_error']])),
            onset={kind: {'gain_delta_db': distribution([v['gain_delta_db'] for r in items for v in r['onsets'] if v['kind']==kind]),
                         'gain_delta': distribution([v['gain_delta'] for r in items for v in r['onsets'] if v['kind']==kind])}
                   for kind in ('bystander_first', 'reentry')})
    return {'all': summarize(rows), 'by_bucket_and_shape': {
        f'{seconds:g}s/shape_{shape}': summarize([r for r in rows if r['seconds']==seconds and r['shape']==shape])
        for seconds,shape in sorted({(r['seconds'],r['shape']) for r in rows})}}


@torch.no_grad()
def evaluate_session_manifest(model, manifest_path, *, device='cpu', min_distance_gap=.25,
                              min_turn_frames=1, presence_delay_frames=0,
                              waveform_delay_samples=0, onset_window_seconds=1., bounded_limit=None,
                              pair_selection="cross_role", target_role=1):
    """Reusable callback entry point; preserves caller model training/eval mode.

    Reference = fresh forward beginning exactly at the turn onset, with identical
    current mixture and clean target. Delta is contextual minus fresh gain, so a
    negative value means prior context reduced target-correlated output. Reentry
    means a later user turn after >= 1 second without user turn activity.
    """
    path = Path(manifest_path); manifest = json.loads(path.read_text())
    assert_disjoint(manifest['speaker_pools']['train'], manifest['speaker_pools']['validation'])
    sr, fps = manifest['sample_rate'], manifest['fps']
    if waveform_delay_samples < 0 or onset_window_seconds <= 0:
        raise ValueError('nonnegative waveform delay and positive onset window required')
    was_training = model.training
    model.eval(); results = []
    try:
        for entry in manifest['rows']:
            row_path = path.parent/entry['file']
            if digest(row_path) != entry['sha256']:
                raise ValueError(f'corrupt fixed validation artifact: {row_path}')
            batch = torch.load(row_path, map_location='cpu', weights_only=True)
            x, target = batch['noisy_speech'].to(device), batch['clean_speech'].reshape(-1)
            output = model(x).detach().cpu().reshape(-1)
            proximity = getattr(model.backbone, 'last_proximity', None)
            vad = getattr(model.backbone, 'last_vad_logits', None)
            if proximity is None or vad is None:
                raise ValueError('session evaluation requires ProximityHead and VAD head readouts')
            proximity, vad = proximity.detach().cpu().clone(), vad.detach().cpu().clone()
            paired = batch['paired_view']
            if not torch.equal(paired['row_source_id'], batch['row_source_id'][paired['source_indices']]):
                raise ValueError('paired source identity mismatch')
            if torch.equal(paired['turn_chain'], batch['turn_chain'][paired['source_indices']]):
                raise ValueError('paired view must have a different chain draw id')
            model(paired['noisy_speech'].to(device))
            q = proximity_metrics(proximity, batch, model.backbone.last_proximity.detach().cpu(),
                                  min_distance_gap=min_distance_gap, min_turn_frames=min_turn_frames, bounded_limit=bounded_limit,
                                  pair_selection=pair_selection)
            p = presence_metrics(vad, batch['vad_target'], fps=fps, delay_frames=presence_delay_frames)
            ids, roles = _array(batch['turn_id']), _array(batch['turn_role'])
            onsets, previous_user_end = [], None
            for k, role in enumerate(roles):
                hit = np.flatnonzero(ids == k+1)
                if role != target_role or not hit.size:
                    continue
                start, end = int(hit[0]), int(hit[-1]+1)
                prior_bystander = any(roles[j-1] > 0 and roles[j-1] != target_role for j in np.unique(ids[:start]) if j > 0)
                kind = 'bystander_first' if previous_user_end is None and prior_bystander else (
                    'reentry' if previous_user_end is not None and (start-previous_user_end)/fps >= 1 else None)
                previous_user_end = end
                if kind is None:
                    continue
                a = int(round(start*sr/fps)); b = min(int(round(end*sr/fps)), a+int(onset_window_seconds*sr))
                fresh = model(x[..., a:]).detach().cpu().reshape(-1)
                n = min(b-a, output.numel()-a-waveform_delay_samples, fresh.numel()-waveform_delay_samples)
                if n <= 0:
                    continue
                contextual = target_projection(output[a+waveform_delay_samples:a+waveform_delay_samples+n], target[a:a+n])
                reference = target_projection(fresh[waveform_delay_samples:waveform_delay_samples+n], target[a:a+n])
                if contextual is not None and reference is not None:
                    onsets.append(dict(kind=kind, onset_seconds=a/sr, samples=n, contextual=contextual, reference=reference,
                                       gain_delta=contextual['gain']-reference['gain'], gain_delta_db=contextual['gain_db']-reference['gain_db']))
            results.append({**entry, 'presence': p, 'proximity': q, 'onsets': onsets})
    finally:
        model.train(was_training)
    return {'manifest_sha256': digest(path), 'alignment': {'presence_delay_frames': presence_delay_frames,
            'waveform_delay_samples': waveform_delay_samples}, 'min_distance_gap_m': min_distance_gap,
            'min_turn_frames': min_turn_frames, 'onset_window_seconds': onset_window_seconds,
            'bounded_limit': bounded_limit, 'pair_selection': pair_selection, 'target_role': target_role, 'summary': summarize_rows(results), 'rows': results}
