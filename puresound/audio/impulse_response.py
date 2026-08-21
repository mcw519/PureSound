from typing import Optional

import torch
import torchaudio

from puresound.audio.rir.metrics import compute_drr_db as _compute_drr_db
from puresound.utils import fftconvolve


def compute_drr_db(
    rir: torch.Tensor,
    sample_rate: int,
    direct_window_ms: float = 2.5,
) -> float:
    """Direct-to-reverberant ratio (dB) of an RIR.

    Direct path = energy in ``[peak, peak + direct_window_ms]``; everything
    after that window is treated as the reverberant tail. Returns ``+inf``
    when the tail carries no energy (e.g. anechoic or trimmed RIR).
    """
    return _compute_drr_db(
        rir,
        sample_rate=sample_rate,
        direct_window_ms=direct_window_ms,
    )


def wav_apply_rir(
    wav: torch.Tensor,
    impaulse: torch.Tensor,
    sample_rate: int,
    rir_mode: str = "full",
):
    """
    Simulate reverberation data by convolue RIR in waveform.\n

    Args:
        wav: input waveform tensor with time dimension in the last tensor shape, i.e., [..., L]
        impaulse: rir tensor with shpae as  [chaneels, rir length]
        sample_rate: speech sample rate
        rir_mode: select in ["full", "direct", "early"]

    Returns:
        waveform has been convolved with RIR
    """
    rir_mode = rir_mode.lower()
    wav_ch, _ = wav.shape
    rir_ch, _ = impaulse.shape

    assert rir_mode in ["full", "direct", "early"]

    if rir_mode == "full":
        pass

    elif rir_mode == "direct":
        peak_idx = impaulse.argmax().item()
        direct_range = peak_idx + int(sample_rate * 0.006)  # 6ms range
        impaulse = impaulse[:, : int(direct_range)]

    elif rir_mode == "early":
        peak_idx = impaulse.argmax().item()
        early_range = peak_idx + int(sample_rate * 0.05)  # 50ms range
        impaulse = impaulse[:, : int(early_range)]

    # Peak-normalize so convolved output stays at ~ input signal level.
    # Slicing for early/direct preserves the direct-path peak, so all three
    # rir_modes get the same scale factor for the same source RIR — keeping
    # clean_target (early) and noisy_speech (full) at the same direct-path
    # level (only late-reverb energy differs).
    #
    # SIDE EFFECT (level-cue removal): bank RIRs carry a 1/r distance gain on
    # disk with inter-channel ratios intact, but this per-channel normalization
    # discards them — mixtures do NOT inherit a distance level law; only DRR /
    # decay shape / spectral tilt distinguish near from far. The recipes' SIR
    # handling assumes this. Change it and every level-related assumption
    # (mix_mode 'physical', hard-SIR ranges) changes with it.
    peak = impaulse.abs().max()
    if peak > 1e-12:
        impaulse = impaulse / peak

    out = []
    if rir_ch == 1:
        for i in range(wav_ch):
            tmp_wav = fftconvolve(wav[i].view(1, -1), impaulse, mode="full")
            propagation_delays = impaulse.abs().argmax(dim=-1, keepdim=False)[0]
            tmp_wav = tmp_wav[
                ..., propagation_delays : propagation_delays + wav.shape[-1]
            ]
            out.append(tmp_wav)

    else:
        assert wav.shape[0] == 1, "when rir chaneels not equal to 1 (ex: mic array inpaulse), wav must be single channel case."
        for i in range(rir_ch):
            tmp_wav = fftconvolve(wav, impaulse[i].view(1, -1), mode="full")
            propagation_delays = impaulse.abs().argmax(dim=-1, keepdim=False)[0]
            tmp_wav = tmp_wav[
                ..., propagation_delays : propagation_delays + wav.shape[-1]
            ]
            out.append(tmp_wav)

    out = torch.cat(out, dim=0)
    assert wav.shape[-1] == out.shape[-1]

    return out


def rand_add_2nd_filter_response(
    wav: torch.Tensor,
    a: Optional[torch.Tensor] = None,
    b: Optional[torch.Tensor] = None,
):
    """
    Reference:
        [1] A Hybrid DSP/Deep Learning Approach to Real-Time Full-Band Speech Enhancement
    """
    if a is None or b is None:
        r = torch.Tensor(4).uniform_(-3 / 8, 3 / 8)
        a = torch.Tensor([1, r[0], r[1]])
        b = torch.Tensor([1, r[2], r[3]])

    # clamp=False: this models a transducer's frequency response, and a
    # frequency response is linear. `lfilter` hard-clips to [-1, 1] by default,
    # which on a hot mixture is a waveshaper the recipe never asked for -- and
    # one that hits the mixture without touching the quieter target it is scored
    # against. `puresound.audio.dsp.apply_linear` is the same guarantee for the
    # backends that expose no such flag.
    wav = torchaudio.functional.lfilter(
        wav, a_coeffs=a.to(wav.device), b_coeffs=b.to(wav.device), clamp=False
    )

    return wav, a, b


def smear_direct_arrival(
    impaulse: torch.Tensor,
    sample_rate: int,
    *,
    smear_ms: float,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Scramble the structure just after the direct arrival, keeping everything else.

    Where the cue lives was measured window by window
    (`egs/voice_isolate/benchmarks/probes/dist_cue_anatomy_README.md`):
    noise-replacing the first 2.5 ms keeps 45% of the near/far separation against
    60-62% for the early reflections or the late tail, a 5 ms smear leaves 1%,
    and splitting the window from first principles puts it in the fine TIMING --
    destroying timing alone keeps 35%, flattening the spectrum keeps 81%. One
    cue, and reverberation masks it -- which is the mechanism behind deletion
    rising monotonically with RT60.

    This removes it on purpose, on a fraction of rows, so the model cannot rely
    on it alone. Whether a substitute is then learnable is an open question, and
    the same study is the caution: timing and spectrum are read jointly, not
    summed (destroying both keeps 54% -- strictly more damage than timing alone,
    yet a larger surviving gap, so the two readings partially cancel). Spectral
    tilt (centroid 351 Hz near against 620 Hz far, unmasked by reverberation) is
    a candidate substitute, not an established one -- which is why this knob
    ships dark: no recipe sets it until a training run is worth that gamble.

    What is preserved, so the manipulation is about TIMING and nothing else:

    * **the peak index**, because `wav_apply_rir` reads `argmax` for both the
      window cuts and the propagation-delay alignment -- move it and every
      downstream offset moves with it;
    * **the window's energy**, restored after smearing, so this is not a level
      change (the pipeline peak-normalises RIRs anyway, deliberately removing the
      1/r level cue);
    * **the late tail**, untouched past ``smear_ms``.

    Applied to the RIR before `wav_apply_rir` slices it, so the ``full`` mixture
    and the ``early`` target inherit the same smear from the same impulse -- the
    target stays the near component of the smeared mixture.
    """
    if smear_ms <= 0.0:
        return impaulse
    n = int(round(smear_ms * sample_rate / 1000.0))
    if n < 2:
        return impaulse
    out = impaulse.clone()
    for ch in range(out.shape[0]):
        h = out[ch]
        peak = int(h.abs().argmax().item())
        end = min(h.shape[-1], peak + n)
        if end - peak < 2:
            continue
        window = h[peak:end]
        energy = window.pow(2).sum()
        if float(energy) <= 0.0:
            continue
        # A random unit-energy kernel spreads the window in time. Causal, so
        # nothing arrives before the direct path did.
        kernel = torch.randn(end - peak, generator=generator,
                             dtype=h.dtype, device=h.device)
        kernel = kernel / kernel.pow(2).sum().sqrt().clamp_min(1e-12)
        smeared = torch.nn.functional.conv1d(
            torch.nn.functional.pad(window.view(1, 1, -1), (kernel.shape[-1] - 1, 0)),
            kernel.flip(-1).view(1, 1, -1),
        ).view(-1)
        # Restore the peak's dominance BEFORE renormalising, not after: the smear
        # can leave a later sample larger, which would move argmax and silently
        # shift every window cut downstream -- but doing the fix afterwards adds
        # energy back and the window no longer matches what it started with.
        biggest = smeared.abs().max()
        if float(smeared[0].abs()) < float(biggest):
            sign = torch.sign(smeared[0]) if float(smeared[0]) != 0.0 else 1.0
            smeared[0] = sign * biggest * 1.001
        smeared = smeared * (energy / smeared.pow(2).sum().clamp_min(1e-12)).sqrt()
        h[peak:end] = smeared
    return out
