import math
import random
from typing import Callable, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
import torchaudio


#: Digital full scale. The one level in this package that means something
#: physical: the point where an analogue signal becomes samples.
FULL_SCALE = 1.0

#: How close to full scale counts as "the backend saturated". A backend that
#: clipped leaves samples pinned at exactly ±1; real audio lands there only by
#: coincidence, and a false positive costs one extra escalation, nothing more.
_SATURATION_EPS = 1e-6

#: The level a linear backend is handed by :func:`apply_linear`. 0.5 leaves 6 dB
#: for the operator's own gain, which covers every filter in this package -- the
#: random 2nd-order response peaks at +3.5 dB median -- and the escalation loop
#: covers the tail (that response reaches +16 dB at worst).
_LINEAR_BACKEND_HEADROOM = 0.5


def apply_linear(
    fn: Callable[[torch.Tensor], torch.Tensor],
    wav: torch.Tensor,
    *,
    headroom: float = _LINEAR_BACKEND_HEADROOM,
    max_escalations: int = 8,
) -> torch.Tensor:
    """Run a linear DSP backend at a level its implicit clipper cannot reach.

    Every backend this package filters and gains with saturates at digital full
    scale: ``torchaudio.functional.lfilter`` clamps to [-1, 1] unless told not
    to, every ``biquad`` is built on it, and every sox effect round-trips
    through a fixed-point sample format. So a stage that models a *linear*
    operator -- a microphone's frequency response, a rumble filter, a preamp
    gain, a resampler -- silently stops being one the moment the signal handed
    to it is hot. That is not the microphone doing something; it is a library
    default, and it fires on real recipes: measured over the shipped
    voice-isolation recipe it hit 28% of transducer-response calls and 19% of
    gain calls.

    It matters more than the distortion itself. These stages are applied to a
    mixture *and* to the clean target it is scored against, with the same
    parameters, precisely so the pair keeps its level relationship. A ceiling at
    a fixed absolute level breaks that: the mixture is the louder of the two, so
    it is the one that gets squashed while the target passes through untouched,
    and the mixture stops being the sum of its parts at the SIR the recipe
    asked for.

    The fix is the definition of linearity. For a linear operator ``H`` and any
    scalar ``a > 0``, ``H(x) == H(a * x) / a``. So scale into the backend's
    legal range, apply, and scale back: the result is the operator's true
    output, and the saturation never fires. If the backend saturated anyway --
    an operator with more gain than the headroom allowed for -- back off and
    retry rather than return a number that is quietly wrong.

    Use this wherever the backend gives no way to turn its ceiling off. Where it
    does, say so directly instead: ``lfilter(..., clamp=False)`` is exact and
    costs nothing.

    ``fn`` must not consume randomness. An escalation calls it again, which
    would shift the RNG stream and break the seeded-item contract the datasets
    rely on. Every current caller is a pure filter or gain.

    Args:
        fn: the backend, as a one-argument tensor -> tensor call.
        wav: waveform, ``[..., L]``, at whatever level the chain has it.
        headroom: peak level handed to the backend on the first attempt.
        max_escalations: attempts before giving up; each divides by 8.

    Returns:
        ``fn(wav)`` as the linear operator it models, at the input's own level.
    """
    peak = float(wav.abs().amax()) if wav.numel() else 0.0
    if not math.isfinite(peak) or peak <= 0.0:
        # Digital silence, or already broken. Scaling is undefined and there is
        # no headroom question to answer -- hand it straight to the backend.
        return fn(wav)

    scale = headroom / peak
    for _ in range(max_escalations):
        out = fn(wav * scale)
        if not bool((out.abs() >= FULL_SCALE - _SATURATION_EPS).any()):
            return out / scale
        scale /= 8.0

    raise RuntimeError(
        f"{getattr(fn, '__name__', fn)} still saturates after {max_escalations} "
        f"escalations from headroom {headroom}: its gain is beyond anything a "
        "linear stage in this chain should have. Check the operator, not this."
    )


def wav_resampling(
    wav: torch.Tensor,
    origin_sr: int,
    target_sr: int,
    backend: str = "sox",
    torch_backend_params: Optional[Dict] = None,
):
    """
    Audio sample rate resaple.

    Args:
        wav: The waveform used for computing amplitude. Shape should be [..., L]
        origin_sr: original wav's sample rate
        tartget_sr: target sample rate
        backend: choose in ["sox", "torchaudio"]
        torch_backend_params: specific the torchaudio setting

    Returns:
        resampling wav and its target sample rate
    """
    backend = backend.lower()
    assert backend in ["sox", "torchaudio"]

    if origin_sr == target_sr:
        if backend == "torchaudio":
            return wav, target_sr, torch_backend_params or {}
        return wav, target_sr

    if backend == "torchaudio" or not hasattr(torchaudio, "sox_effects"):
        """downsample and upsample back by TorchAudio"""
        lp_width = None
        rolloff = None
        window = None

        if torch_backend_params is not None:
            lp_width = torch_backend_params["lp_width"]
            rolloff = torch_backend_params["rolloff"]
            window = torch_backend_params["window"]

        if lp_width is None:
            lp_width = random.choice((6, 16, 32, 64, 128))

        if rolloff is None:
            rolloff = random.uniform(0.8, 0.99)

        if window is None:
            window = random.choice(("sinc_interp_hann", "sinc_interp_kaiser"))

        wav = torchaudio.transforms.Resample(
            orig_freq=origin_sr,
            new_freq=target_sr,
            lowpass_filter_width=lp_width,
            resampling_method=window,
            rolloff=rolloff,
        )(wav)

        torch_backend_params = {
            "lp_width": lp_width,
            "rolloff": rolloff,
            "window": window,
        }

        if backend == "torchaudio":
            return wav, target_sr, torch_backend_params
        return wav, target_sr

    else:
        """downsample and upsample back by Sox command"""
        effects1 = [
            ["rate", str(target_sr)],
        ]
        wav = apply_linear(
            lambda w: torchaudio.sox_effects.apply_effects_tensor(
                w, origin_sr, effects1
            )[0],
            wav,
        )

        return wav, target_sr


def get_biquad_params(
    gain_dB: float,
    cutoff_freq: float,
    q_factor: float,
    sample_rate: float,
    filter_type: str,
):
    """
    Use design parameters to generate coefficients for a specific filter type.

    Args:
        gain_dB (float): Shelving filter gain in dB
        cutoff_freq (float): Cutoff frequency in Hz.
        q_factor (float): Q factor.
        sample_rate (float): Sample rate in Hz.
        filter_type (str): Filter type. One of ["high_shelf", "low_shelf", "peaking", "lpf", "hpf", "bpf", "notch"]

    Returns:
        b : Numerator filter coefficients stored as [b0, b1, b2]
        a : Denominator filter coefficients stored as [a0, a1, a2]
    """
    filter_type = filter_type.lower()
    assert filter_type in [
        "high_shelf",
        "low_shelf",
        "peaking",
        "lpf",
        "hpf",
        "bpf",
        "notch",
    ]

    A = 10 ** (gain_dB / 40.0)  # log to linear gain
    w0 = 2.0 * np.pi * (cutoff_freq / sample_rate)
    alpha = np.sin(w0) / (2.0 * q_factor)

    sin_w0 = np.sin(w0)
    cos_w0 = np.cos(w0)
    sqrt_A = np.sqrt(A)

    if filter_type == "high_shelf":
        b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * sqrt_A * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
        b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * sqrt_A * alpha)
        a0 = (A + 1) - (A - 1) * cos_w0 + 2 * sqrt_A * alpha
        a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
        a2 = (A + 1) - (A - 1) * cos_w0 - 2 * sqrt_A * alpha
    elif filter_type == "low_shelf":
        b0 = A * ((A + 1) - (A - 1) * cos_w0 + 2 * sqrt_A * alpha)
        b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
        b2 = A * ((A + 1) - (A - 1) * cos_w0 - 2 * sqrt_A * alpha)
        a0 = (A + 1) + (A - 1) * cos_w0 + 2 * sqrt_A * alpha
        a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
        a2 = (A + 1) + (A - 1) * cos_w0 - 2 * sqrt_A * alpha
    elif filter_type == "peaking":
        b0 = 1 + alpha * A
        b1 = -2 * cos_w0
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * cos_w0
        a2 = 1 - alpha / A
    elif filter_type == "lpf":
        b0 = (1 - cos_w0) / 2
        b1 = 1 - cos_w0
        b2 = (1 - cos_w0) / 2
        a0 = 1 + alpha
        a1 = -2 * cos_w0
        a2 = 1 - alpha
    elif filter_type == "hpf":
        b0 = (1 + cos_w0) / 2
        b1 = -(1 + cos_w0)
        b2 = (1 + cos_w0) / 2
        a0 = 1 + alpha
        a1 = -2 * cos_w0
        a2 = 1 - alpha
    elif filter_type == "bpf":
        b0 = sin_w0 / 2
        b1 = 0
        b2 = -(sin_w0 / 2)
        a0 = 1 + alpha
        a1 = -2 * cos_w0
        a2 = 1 - alpha
    elif filter_type == "notch":
        b0 = 1
        b1 = -2 * cos_w0
        b2 = 1
        a0 = 1 + alpha
        a1 = -2 * cos_w0
        a2 = 1 - alpha

    b = np.array([b0, b1, b2]) / a0
    a = np.array([a0, a1, a2]) / a0
    return b, a


def wav_apply_biquad_filter(
    wav: torch.Tensor, b_coeff: np.ndarray, a_coeff: np.ndarray
):
    """
    Applies the Biquad-Filter

    Args:
        wav: The waveform used for computing amplitude. Shape should be [..., L]
    """
    proc_wav = wav.clone()
    if isinstance(wav, torch.Tensor):
        proc_wav = proc_wav.numpy()

    if proc_wav.ndim == 1:
        proc_wav = np.expand_dims(proc_wav, axis=0)

    ch = proc_wav.shape[0]

    for c in range(ch):
        proc_wav[c] = scipy.signal.lfilter(b_coeff, a_coeff, proc_wav[c])

    return torch.from_numpy(proc_wav)


class ParametricEQ:
    """
    Parametric EQ by series Biquad Filter

    Args:
        sample_rate: waveform sampling rate
        eq_band_gain: series of gain at each band
        eq_band_cutoff: series of cutoff frequency at each band
        eq_band_q_factor: series of Q factor at each band filter
        low_shelf_gain_dB: gain of low shelf filter
        low_shelf_cutoff_freq: cutoff frequency of low shelf filter
        low_shelf_q_factor: Q factor of low shelf filter
        high_shelf_gain_dB: gain of high shelf filter
        high_shelf_cutoff_freq: cutoff frequency of high shelf filter
        high_shelf_q_factor: Q factor of high shelf filter
        dtype: default data type in numpy
    """

    def __init__(
        self,
        sample_rate: float,
        eq_band_gain: Tuple[float],
        eq_band_cutoff: Tuple[float],
        eq_band_q_factor: Tuple[float],
        low_shelf_gain_dB: float = 0.0,
        low_shelf_cutoff_freq: float = 80,
        low_shelf_q_factor: float = 0.707,
        high_shelf_gain_dB: float = 0.0,
        high_shelf_cutoff_freq: float = 1000,
        high_shelf_q_factor: float = 0.707,
        dtype=np.float32,
    ):
        assert len(eq_band_gain) == len(eq_band_cutoff) == len(eq_band_q_factor)
        self.dtype = dtype
        self.sr = sample_rate

        self.b_list = []
        self.a_list = []

        b, a = get_biquad_params(
            gain_dB=low_shelf_gain_dB,
            cutoff_freq=low_shelf_cutoff_freq,
            q_factor=low_shelf_q_factor,
            sample_rate=sample_rate,
            filter_type="low_shelf",
        )
        self.b_list.append(b)
        self.a_list.append(a)

        for i in range(len(eq_band_gain)):
            b, a = get_biquad_params(
                gain_dB=eq_band_gain[i],
                cutoff_freq=eq_band_cutoff[i],
                q_factor=eq_band_q_factor[i],
                sample_rate=sample_rate,
                filter_type="peaking",
            )
            self.b_list.append(b)
            self.a_list.append(a)

        b, a = get_biquad_params(
            gain_dB=high_shelf_gain_dB,
            cutoff_freq=high_shelf_cutoff_freq,
            q_factor=high_shelf_q_factor,
            sample_rate=sample_rate,
            filter_type="high_shelf",
        )
        self.b_list.append(b)
        self.a_list.append(a)

        self.n_eq = len(self.a_list)

    def forward(self, wav: torch.Tensor):
        """
        Applies series of Filters

        Args:
            wav: The waveform used for computing amplitude. Shape should be [..., L]

        Returns:
            filtered wav has same shape of input
        """
        for i in range(self.n_eq):
            wav = wav_apply_biquad_filter(
                wav=wav, b_coeff=self.b_list[i], a_coeff=self.a_list[i]
            )

        return wav

    def plot_eq(self, savefig: Optional[str] = None):
        """Plotting the EQ curves"""
        if self.sr == 16000:
            nfft = 512
        elif self.sr == 32000:
            nfft = 1024
        else:
            raise ValueError

        b = torch.stack([torch.from_numpy(x) for x in self.b_list])
        a = torch.stack([torch.from_numpy(x) for x in self.a_list])

        B = torch.fft.rfft(b, nfft)
        A = torch.fft.rfft(a, nfft)

        H = B / A
        H = torch.prod(H, dim=0).view(-1)
        H = H.abs()
        faxis = torch.linspace(0, self.sr // 2, steps=(nfft // 2) + 1)
        plt.plot(faxis, H)
        plt.xlabel("Hz")
        plt.title("Parametric EQ")
        if savefig is not None:
            plt.savefig(savefig)


def compressor_gain(
    wav: torch.Tensor,
    sample_rate: int,
    *,
    threshold_db: float,
    ratio: float,
    attack_ms: float = 5.0,
    release_ms: float = 120.0,
    makeup: bool = True,
) -> torch.Tensor:
    """The gain curve a broadcast-style compressor would apply, as a [1, T] tensor.

    Returned rather than applied, because the caller has to put the SAME curve on
    the mixture and on the target. A compressor is a time-varying GAIN, so
    ``g * (near + far) == g * near + g * far`` and superposition survives -- which
    is the whole reason this is not the ``|x|^p`` waveshaper in
    `Augmentor.apply_media_coloring`. That one is a distortion: it is the right
    model for a TV loudspeaker in the room, and the wrong one for "the recording
    was compressed", because no per-source decomposition of it exists and the
    target could only follow by being separately distorted into something that is
    no longer the mixture's near component.

    Detector is a peak follower on the absolute signal with asymmetric one-pole
    smoothing, the usual shape: fast attack so a transient is caught, slow
    release so the gain does not pump between syllables. Gain reduction is
    computed in dB above ``threshold_db`` at ``ratio``, then optionally makeup-
    normalised so the curve has unit mean -- without that, compression and a
    level change are the same augmentation and a model could learn either.
    """
    if ratio < 1.0:
        raise ValueError(f"ratio must be >= 1 (1.0 = no compression), got {ratio}")
    if not (attack_ms > 0.0 and release_ms > 0.0):
        raise ValueError(f"attack/release must be > 0 ms, got {attack_ms}/{release_ms}")
    flat = wav.reshape(-1).abs().float()
    a_att = 1.0 - math.exp(-1000.0 / (attack_ms * sample_rate))
    a_rel = 1.0 - math.exp(-1000.0 / (release_ms * sample_rate))
    # max(fast pole, slow pole) -- a standard asymmetric follower, and NOT an
    # approximation of the per-sample `coefficient switches on x > env` loop: it
    # correlates 0.93 with it, not 1.0. It is the same SHAPE, which is what a
    # detector has to be: on an onset the fast pole is higher, so the envelope
    # rises at the attack rate; through a decay the slow pole is higher, so it
    # falls at the release rate. Chosen because the per-sample loop costs 2.4 s
    # for six seconds of audio -- 600+ minutes of dataloader per epoch -- and
    # this costs 12 ms. Which detector a compressor uses is a design choice; the
    # properties the augmentation needs are asserted in the tests instead.
    def one_pole(x, a):
        return torchaudio.functional.lfilter(
            x, a_coeffs=x.new_tensor([1.0, -(1.0 - a)]),
            b_coeffs=x.new_tensor([a, 0.0]), clamp=False,
        )

    fast = one_pole(flat, a_att)
    slow = one_pole(flat, a_rel)
    env = torch.maximum(fast, slow)
    env_db = 20.0 * torch.log10(env.clamp_min(1e-8))
    over = (env_db - threshold_db).clamp_min(0.0)
    gain_db = -over * (1.0 - 1.0 / ratio)
    gain = torch.pow(10.0, gain_db / 20.0)
    if makeup:
        gain = gain / gain.mean().clamp_min(1e-8)
    return gain.reshape(1, -1).to(wav.dtype)
