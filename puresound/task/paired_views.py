"""Task-independent auxiliary capture views and source-preserving collation.

Auxiliary views are nested, never appended to the primary batch. Consumers opt
into their consistency objective explicitly; ordinary supervised losses see B
independent sources exactly as before.
"""
from typing import Dict, Sequence

import torch
from torch.nn.utils.rnn import pad_sequence


def apply_chain_views(chain, noisy, target, *, sample_rate, sample_length, probability=0.0):
    """Run the primary chain and optionally another draw on identical inputs.

    Clone before potentially in-place stages. Disabled pairing performs no
    clones or random draws and preserves the primary chain's RNG position.
    Equal cropped waveforms do not constitute an effective consistency pair.
    """
    inputs = (noisy.clone(), target.clone()) if probability > 0 else None
    primary = chain.apply(noisy, target, sample_rate=sample_rate)
    auxiliary = None
    if probability > 0 and torch.rand(1).item() < probability:
        candidate = chain.apply(*inputs, sample_rate=sample_rate)
        if not torch.equal(candidate.noisy[..., :sample_length],
                           primary.noisy[..., :sample_length]):
            auxiliary = candidate
    return primary, auxiliary


def collate_paired_views(batch: Sequence[Dict], out: Dict) -> Dict:
    """Collate mono auxiliary waveforms and map them to primary source indices.

    No task labels are required. Optional ``turn_chain`` provenance is padded
    to the primary batch turn width when provided by a task.
    """
    paired = [(i, row['paired_view']) for i, row in enumerate(batch) if 'paired_view' in row]
    if not paired:
        return out
    out['row_source_id'] = torch.tensor([
        int(row.get('row_source_id', -1)) for row in batch
    ], dtype=torch.long)
    view = {
        'source_indices': torch.tensor([i for i, _ in paired], dtype=torch.long),
        'row_source_id': torch.tensor([int(row['row_source_id']) for _, row in paired], dtype=torch.long),
    }
    for key in ('noisy_speech', 'clean_speech'):
        padded = pad_sequence([row[key].reshape(-1) for _, row in paired], batch_first=True)
        # Label-only callers can collate provenance before waveforms exist.
        width = out.get(key, padded).shape[-1]
        if padded.shape[-1] > width:
            raise ValueError('paired waveform exceeds primary batch length')
        view[key] = torch.nn.functional.pad(padded, (0, width - padded.shape[-1]))
    if all('turn_chain' in row for _, row in paired):
        width = max([row.get('turn_chain', torch.empty(0)).numel() for row in batch]
                    + [row['turn_chain'].numel() for _, row in paired])
        chains = torch.zeros(len(paired), width, dtype=torch.long)
        for i, (_, row) in enumerate(paired):
            chains[i, :row['turn_chain'].numel()] = row['turn_chain'].reshape(-1)
        view['turn_chain'] = chains
    out['paired_view'] = view
    return out
