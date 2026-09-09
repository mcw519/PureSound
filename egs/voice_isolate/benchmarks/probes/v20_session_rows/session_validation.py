"""Materialize fixed session validation, then evaluate without training.

Run from egs/voice_isolate so recipe corpus/augmentation paths resolve as training does.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from puresound.evaluation.session_validation import evaluate_session_manifest, digest, assert_disjoint, persist_batch, require_checkpoint_heads


def materialize(recipe_path, output, *, seed=20260905, rows_per_bucket=20, lengths=(12., 30.)):
    """Use the production dataset on an evaluation-only recipe copy; freeze tensors."""
    from puresound.config import load_recipe
    from puresound.config.base import with_overrides
    from puresound.dataset.parser import MetafileParser
    from puresound.task.voice_isolation import VoiceIsolationDataset, VoiceIsolationCollateFunc
    recipe_path, output = Path(recipe_path).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'refusing to overwrite materialized evaluation: {output}')
    if rows_per_bucket < 1 or not lengths or any(s not in (12., 30.) for s in lengths):
        raise ValueError('positive rows_per_bucket and 12/30 second buckets required')
    recipe = load_recipe(str(recipe_path), expected_task='voice_isolation', expected_purpose='train')
    corpus = recipe.dataset
    pools = {}
    for kind, path in [('train', corpus.train_metafile), ('validation', corpus.valid_metafile)]:
        pools[kind] = sorted(MetafileParser.read_from_metafile(path, use_speaker_as_key=True))
    assert_disjoint(pools['train'], pools['validation'])
    if recipe.augmentation_session_rows is None:
        raise ValueError('recipe must define augmentation_session_rows')
    evaluation = with_overrides(recipe, augmentation_session_rows=dict(
        enabled=True, prob=1., min_seconds=12., pair_prob=0.,
        paired_view_prob=1., paired_view_min_seconds=12.))
    dataset = VoiceIsolationDataset(metafile_path=corpus.valid_metafile,
        min_utt_length_in_seconds=corpus.filter_min_utterance_length,
        min_utts_in_each_speaker=corpus.filter_min_utterance_per_speaker,
        target_sr=corpus.target_sample_rate, training_sample_length_in_seconds=12.,
        audio_gain_normalized_to=corpus.gain_normalized_to, dataset_role='validation',
        pipeline_role=corpus.validation_pipeline_role, **evaluation.augmentation_kwargs())
    speakers = sorted(dataset.total_spks)
    assert_disjoint(pools['train'], speakers)
    sr = corpus.target_sample_rate
    if not sr:
        raise ValueError('fixed session validation requires target_sample_rate')
    label_args = evaluation.vad_label.args
    fps = sr / label_args.get('hop_length', evaluation.vad_label.hop_length)
    output.mkdir(parents=True, exist_ok=True)
    (output/'recipe.yaml').write_bytes(recipe_path.read_bytes())
    rows = []
    for seconds in lengths:
        for _ in range(rows_per_bucket):
            i = len(rows)
            for attempt in range(50):
                row_seed = seed+i*50+attempt
                batch = VoiceIsolationCollateFunc()([dataset[(speakers[i % len(speakers)], sr, row_seed, seconds)]])
                if float(batch['session_row'][0]) == 1 and 'paired_view' in batch:
                    break
            else:
                raise ValueError('no nonidentical paired chain view after 50 draws; enable stochastic device chain')
            rows.append(persist_batch(output, batch, row_id=i, seed=row_seed, seconds=seconds))
            rows[-1]["chain_draw_rejections"] = attempt
    manifest = dict(schema_version=1, seed=seed, sample_rate=sr, fps=fps,
                    generator='puresound.task.voice_isolation.VoiceIsolationDataset',
                    selection_protocol='reject identical second-chain waveforms, max 50 deterministic attempts per row',
                    recipe_sha256=digest(recipe_path), evaluation_recipe=evaluation.model_dump(mode='json'),
                    speaker_pools=pools, eligible_validation_speakers=speakers,
                    metafiles={k: {'path': str(Path(p).resolve()), 'sha256': digest(p)} for k,p in
                               [('train', corpus.train_metafile), ('validation', corpus.valid_metafile)]}, rows=rows)
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return output/'manifest.json'



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    make = commands.add_parser('materialize')
    make.add_argument('recipe'); make.add_argument('--out', required=True)
    make.add_argument('--seed', type=int, default=20260905)
    make.add_argument('--rows-per-bucket', type=int, default=20)
    evaluate = commands.add_parser('evaluate')
    evaluate.add_argument('recipe'); evaluate.add_argument('--manifest', required=True)
    evaluate.add_argument('--ckpt', required=True); evaluate.add_argument('--out', required=True)
    evaluate.add_argument('--device', default='cpu')
    evaluate.add_argument('--min-distance-gap', type=float, default=.25)
    evaluate.add_argument('--min-turn-frames', type=int, default=1)
    evaluate.add_argument('--presence-delay-frames', type=int, default=0)
    evaluate.add_argument('--waveform-delay-samples', type=int, default=0)
    evaluate.add_argument('--bounded-limit', type=float, default=None)
    evaluate.add_argument('--pair-selection', choices=['cross_role', 'all'], default='cross_role')
    for command in (make, evaluate):
        command.add_argument('--num-threads', type=int, default=1)
    args = parser.parse_args()
    import torch
    if args.num_threads < 1:
        parser.error('--num-threads must be positive')
    torch.set_num_threads(args.num_threads)
    if args.command == 'materialize':
        print(materialize(args.recipe, args.out, seed=args.seed, rows_per_bucket=args.rows_per_bucket))
        return
    import torch
    from puresound.config import load_recipe
    from puresound.recipes import init_siso_model
    recipe = load_recipe(args.recipe, expected_task='voice_isolation', expected_purpose='train')
    model = init_siso_model(recipe.model)
    state = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    state = state.get('state_dict', state)
    require_checkpoint_heads(model, state)
    model.reload_checkpoint(state, load_loss_func=False)
    model.to(args.device)
    result = evaluate_session_manifest(model, args.manifest, device=args.device,
        min_distance_gap=args.min_distance_gap, min_turn_frames=args.min_turn_frames,
        presence_delay_frames=args.presence_delay_frames, waveform_delay_samples=args.waveform_delay_samples,
        bounded_limit=args.bounded_limit, pair_selection=args.pair_selection)
    result.update(checkpoint=str(Path(args.ckpt).resolve()), checkpoint_sha256=digest(args.ckpt),
                  evaluation_recipe_sha256=digest(args.recipe))
    Path(args.out).write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result['summary'], indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
