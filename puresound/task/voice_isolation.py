from typing import Dict, List, Optional

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from puresound.task.ns import NoiseSuppressionCollateFunc, NoiseSuppressionDataset


VOICE_ISOLATION_SCALAR_KEYS = (
    "foreground_distance",
    "foreground_drr",
    "nearest_interferer_distance",
    "strongest_interferer_drr",
    "drr_gap",
    "rt60",
    "foreground_margin",
    "interferer_margin",
    "boundary_margin",
    "n_interferers",
    "target_absent",
    "distance_gated",
    "target_present",
    "has_background_speech",
    # DISTANCE_PARENT proposal §7: parent composition + mixing diagnostics.
    "near_count",
    "far_count",
    "mix_mode",
    "realized_speech_sir",
    "noise_snr",
    "overlap_fraction",
)

# mix_mode is emitted as a float code so it collates with the other scalars.
MIX_MODE_CODES = {
    "none": 0.0,
    "legacy": 1.0,
    "physical": 2.0,
    "moderate": 3.0,
    "counter_level": 4.0,
}


class VoiceIsolationDataset(NoiseSuppressionDataset):
    """Distance-cued foreground voice isolation dataset.

    This task reuses the dynamic speech/noise/reverb machinery from
    NoiseSuppressionDataset, but exposes the extra labels that are specific to
    foreground voice isolation: distance-query metadata, DRR separability,
    boundary confidence, target-present flags, and background-speech activity.
    """

    def _emit_task_metadata(
        self,
        sample: Dict,
        *,
        query_distance: Optional[float],
        foreground_metadata: Optional[dict],
        interferer_metadata: List[dict],
        target_absent: bool,
        distance_gated: bool,
        background_speech_reference: Optional[torch.Tensor],
        near_count: int = 1,
        far_count: int = 0,
        mix_mode: str = "none",
        realized_speech_sir: float = float("nan"),
        noise_snr: float = float("nan"),
        overlap_fraction: float = float("nan"),
    ) -> None:
        sample.update(
            self._build_voice_isolation_metadata(
                query_distance=query_distance,
                foreground_metadata=foreground_metadata,
                interferer_metadata=interferer_metadata,
                target_absent=target_absent,
                distance_gated=distance_gated,
                background_speech_reference=background_speech_reference,
                near_count=near_count,
                far_count=far_count,
                mix_mode=mix_mode,
                realized_speech_sir=realized_speech_sir,
                noise_snr=noise_snr,
                overlap_fraction=overlap_fraction,
            )
        )

    def _build_voice_isolation_metadata(
        self,
        query_distance: Optional[float],
        foreground_metadata: Optional[dict],
        interferer_metadata: List[dict],
        target_absent: bool,
        distance_gated: bool,
        background_speech_reference: Optional[torch.Tensor],
        near_count: int = 1,
        far_count: int = 0,
        mix_mode: str = "none",
        realized_speech_sir: float = float("nan"),
        noise_snr: float = float("nan"),
        overlap_fraction: float = float("nan"),
    ) -> Dict[str, torch.Tensor]:
        """Create scalar labels useful for distance-cued voice isolation.

        These labels are generated from simulation metadata only. They are not
        inference inputs; auxiliary heads can use them as supervision so the
        model learns explicit distance/DRR confidence instead of relying only on
        the enhancement loss.
        """
        nan = float("nan")

        def _meta_float(meta: Optional[dict], key: str) -> float:
            if not meta:
                return nan
            value = meta.get(key)
            return float(value) if value is not None else nan

        fg_dist = _meta_float(foreground_metadata, "source_receiver_distance")
        fg_drr = _meta_float(foreground_metadata, "drr_db")
        rt60 = _meta_float(foreground_metadata, "rt60")
        itf_distances = [
            float(m["source_receiver_distance"])
            for m in interferer_metadata
            if m.get("source_receiver_distance") is not None
        ]
        itf_drrs = [
            float(m["drr_db"]) for m in interferer_metadata if m.get("drr_db") is not None
        ]
        min_itf_dist = min(itf_distances) if itf_distances else nan
        max_itf_drr = max(itf_drrs) if itf_drrs else nan
        qd = float(query_distance) if query_distance is not None else nan

        fg_margin = qd - fg_dist if not np.isnan(qd) and not np.isnan(fg_dist) else nan
        itf_margin = (
            min_itf_dist - qd
            if not np.isnan(qd) and not np.isnan(min_itf_dist)
            else nan
        )
        margins = [v for v in (abs(fg_margin), abs(itf_margin)) if not np.isnan(v)]
        boundary_margin = min(margins) if margins else nan
        drr_gap = fg_drr - max_itf_drr if not np.isnan(fg_drr) and not np.isnan(max_itf_drr) else nan
        has_background_speech = (
            background_speech_reference is not None
            and background_speech_reference.abs().amax().item() > 0.0
        )
        target_present = not target_absent and not distance_gated

        values = {
            "foreground_distance": fg_dist,
            "foreground_drr": fg_drr,
            "nearest_interferer_distance": min_itf_dist,
            "strongest_interferer_drr": max_itf_drr,
            "drr_gap": drr_gap,
            "rt60": rt60,
            "foreground_margin": fg_margin,
            "interferer_margin": itf_margin,
            "boundary_margin": boundary_margin,
            "n_interferers": float(len(interferer_metadata)),
            "target_absent": float(target_absent),
            "distance_gated": float(distance_gated),
            "target_present": float(target_present),
            "has_background_speech": float(has_background_speech),
            "near_count": float(near_count),
            "far_count": float(far_count),
            "mix_mode": MIX_MODE_CODES.get(mix_mode, float("nan")),
            "realized_speech_sir": float(realized_speech_sir),
            "noise_snr": float(noise_snr),
            "overlap_fraction": float(overlap_fraction),
        }
        return {
            key: torch.tensor(value, dtype=torch.float32)
            for key, value in values.items()
        }


class VoiceIsolationCollateFunc(NoiseSuppressionCollateFunc):
    """Collate waveform batches plus voice-isolation scalar/frame labels."""

    def __call__(self, batch: Dict):
        out = super().__call__(batch)

        for key in VOICE_ISOLATION_SCALAR_KEYS:
            values = [b[key].view(-1) for b in batch if key in b]
            if values:
                out[key] = torch.cat(values, dim=0)

        # Far-parent target waveform (P1): pad like the other waveforms so an
        # optional far decoder loss can read batch["far_target"].
        if any("far_target" in b for b in batch):
            far = [
                (b["far_target"] if "far_target" in b else torch.zeros_like(b["noisy_speech"])).squeeze()
                for b in batch
            ]
            out["far_target"] = pad_sequence(far, batch_first=True)

        has_background_vad = any("background_vad_target" in b for b in batch)
        has_background_vad_ref = any("background_vad_reference" in b for b in batch)
        background_vad = []
        background_vad_ref = []
        for b in batch:
            if has_background_vad:
                if "background_vad_target" not in b:
                    template = b.get("vad_target")
                    if template is None:
                        continue
                    background_vad.append(torch.zeros_like(template).squeeze())
                else:
                    background_vad.append(b["background_vad_target"].squeeze())
            if has_background_vad_ref:
                if "background_vad_reference" not in b:
                    background_vad_ref.append(torch.zeros_like(b["noisy_speech"]).squeeze())
                else:
                    background_vad_ref.append(b["background_vad_reference"].squeeze())

        if background_vad:
            out["background_vad_target"] = pad_sequence(background_vad, batch_first=True)
        if background_vad_ref:
            out["background_vad_reference"] = pad_sequence(
                background_vad_ref,
                batch_first=True,
            )
        return out
