# puresound.nnet.lobe.ssm

繁體中文版本：[ssm.zh-TW.md](ssm.zh-TW.md)

Selective state-space (Mamba / S6) block for streaming enhancement (Gu and Dao,
"Mamba: Linear-Time Sequence Modeling with Selective State Spaces", 2023).

`MambaInter` is a drop-in for the inter(-time) `SingleRNN` of `DPRNNblock2D`:
`[N, D, T]` in, `[N, D, T]` out, causal. The inter path is the only component of
a dual-path model that carries context across time, and what an LSTM learns
there is bounded by the training row length and by how far backpropagation
through time reaches. An SSM keeps a per-frame recurrent form — no look-ahead,
a per-step CPU cost of the same order as the LSTM step — while its
input-dependent decay is built for longer contexts.

## Computation

```
[x', z]      = in_proj(x)                        # D -> 2 * d_inner
u            = SiLU(causal depthwise Conv1d(x')) # kernel d_conv
[dt_r, B, C] = x_proj(u)                         # dt_rank + 2 * d_state
Δ            = softplus(dt_proj(dt_r))           # per channel and frame
A            = -exp(A_log)                       # [d_inner, d_state], negative real
h_t          = exp(Δ_t A) ⊙ h_{t-1} + (Δ_t u_t) B_t
y_t          = <h_t, C_t> + D ⊙ u_t
out          = out_proj(Dropout(y ⊙ SiLU(z)))    # d_inner -> D
```

## Class: `MambaInter`

```python
MambaInter(
    d_model: int,                  # D, the inter path width
    d_state: int = 16,             # SSM state size per channel
    d_conv: int = 4,               # causal depthwise conv kernel
    expand: int = 2,               # d_inner = expand * d_model
    dt_rank: Optional[int] = None, # defaults to ceil(d_model / 16)
    dropout: float = 0.0,          # before out_proj
    dt_min: float = 0.001,         # range of the initial step size Δ
    dt_max: float = 0.1,
    dt_init_floor: float = 1e-4,
    zero_init_out: bool = False,   # start out_proj at zero
)
```

- `forward(x [N, D, T]) -> [N, D, T]`.
- `initial_stream_state(batch, device=None, dtype=None) -> (conv_cache [N, d_inner, d_conv-1], h [N, d_inner, d_state])`;
  the conv cache is in the model dtype, `h` always fp32.
- `step(x_t [N, D], state) -> (y_t [N, D], state)` advances one frame and matches
  the sequential scan exactly.

Initialisation: `A` is S4D-real (`A_log = log(1..d_state)` per channel); the
`dt_proj` bias is set so that `softplus(bias)` is log-uniform in
`[dt_min, dt_max]` (floored at `dt_init_floor`), which sets the initial memory
horizon.

**Parameter budget.** At `d_model` 128 the defaults give about 116k parameters,
against about 99k for `SingleRNN("LSTM", 128, 96)` with its projection, so
replacing one with the other trades capacity like for like.

### Use in DPCRN

```yaml
backbone_args:
  inter_type: lstm+mamba      # lstm | mamba | mamba_context | lstm+mamba
  mamba_args: {d_state: 16, d_conv: 4, expand: 2}
```

`mamba` replaces the inter LSTM; `lstm+mamba` keeps the LSTM and adds
`MambaInter(zero_init_out=True)` as a parallel branch; `mamba_context` runs the
block on perceptual bands. See [DPCRN](../dpcrn.md).

### Design notes

- **`zero_init_out`.** With the output projection at zero the block emits
  exactly 0, so a parallel branch added to a trained LSTM leaves its output
  unchanged at step 0: a warm start carries no re-initialisation, and the SSM's
  inner parameters start receiving gradient only once `out_proj` moves.
- **fp32 state.** `h` is multiplied by `exp(Δ A)` every frame; in bf16 the small
  per-step increments underflow, so `h` and the recurrence run in fp32 whatever
  the autocast mode.

## Execution paths

`forward` picks its scan at run time:

```python
use_kernel = selective_scan_fn is not None and u.is_cuda and not torch.jit.is_tracing()
```

| path | when | notes |
|---|---|---|
| `mamba_ssm`'s fused `selective_scan_fn` | CUDA, kernel importable, not tracing | the training path |
| `_scan_fallback` | CPU, tracing, or no kernel | per-frame Python loop in chunks of `SCAN_CHUNK = 64`; exact |
| `step()` | streaming | one frame, explicit `(conv_cache, h)` state; what the streaming export runs |

All three share one parameter set. `_load_selective_scan` loads only
`mamba_ssm.ops.selective_scan_interface` (not the package, whose generation
utilities pull in specific `transformers` versions) and returns `None` on any
failure, because the fallback is a supported configuration.

**A missing kernel is silent and expensive.** The fallback is correct but much
slower, and almost all of the difference lands in the backward pass. The
streaming export and its real-time-factor check run `step()` per frame, so they
cannot see it; training cost has to be checked separately. If training is
unexpectedly slow, check in the environment the training runs in:

```python
from puresound.nnet.lobe.ssm import selective_scan_fn
print(selective_scan_fn is not None)
```

A different working directory can resolve `mamba_ssm` to a source tree instead
of the installed package and give a different answer.

## Memory and chunking

In training the whole block is wrapped in a checkpoint (the projection
intermediates would otherwise be held for the whole sequence), and on the
fallback path each `SCAN_CHUNK`-frame chunk is checkpointed again, so backward
keeps only chunk boundaries. The fp32 copies of `u`, `Δ`, `B`, `C` and `z` exist
only inside the checkpointed chunk, never at full length. A larger
`SCAN_CHUNK` does not make the fallback faster — the cost is the sequential
loop — and retains a larger autograd graph.

## Rebuilding the kernel against a new PyTorch

The prebuilt `selective_scan_cuda` extension links against PyTorch's internal
C++ ABI, which is not stable across releases. After a torch upgrade the import
can fail with:

```
ImportError: selective_scan_cuda...so: undefined symbol:
  _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_ib
```

Demangled, the extension wants
`c10::cuda::c10_cuda_check_implementation(int, char const*, char const*, int, bool)`
while the installed torch exports
`(int, char const*, char const*, unsigned int, bool, c10::cuda::CUDAErrorLogCapture*)`.
The signature changed; no symlink or `LD_PRELOAD` helps. The fix is to
recompile against the installed torch.

Three things block a plain `pip install mamba-ssm`:

**The PyPI sdist has no CUDA sources.** `csrc/` is absent and the build fails
with `selective_scan.cpp: No such file or directory`. Build from the GitHub tag
of the release you need (`v2.2.4` below):

```bash
git clone --depth 1 --branch v2.2.4 https://github.com/state-spaces/mamba.git
```

**Recent torch headers require C++20**, and mamba's `setup.py` pins
`-std=c++17`:

```
ATen/ATen.h:5: error: C++20 or later compatible compiler is required to use ATen.
```

Change the four `-std=c++17` occurrences in `setup.py` to `-std=c++20`.

**The compiler has to be C++20.** GCC 10 accepts `-std=c++20` but reports
`__cplusplus = 201709L` (the draft), and torch tests for `>= 202002L`, so it
fails the same check. GCC 11+ is required. Where the distribution has nothing
newer, an isolated toolchain leaves the system and the project venv alone:

```bash
conda create -y -p <toolchain-prefix> -c conda-forge 'gxx_linux-64=12' 'gcc_linux-64=12'
```

The GCC version must also be one your CUDA toolkit's `nvcc` accepts as a host
compiler (CUDA 12.3, for example, accepts up to GCC 12); check the toolkit's
release notes before picking it.

Then build **in place**, so nothing is installed until it has been verified:

```bash
cd mamba
export MAMBA_FORCE_BUILD=TRUE                  # never fetch a prebuilt wheel
export TORCH_CUDA_ARCH_LIST=<major.minor>      # your GPU's compute capability only
export CUDA_HOME=<cuda-toolkit-dir>            # toolkit matching torch.version.cuda
export MAX_JOBS=8                              # each nvcc job needs a few GB of RAM
export CC=<toolchain-prefix>/bin/x86_64-conda-linux-gnu-gcc
export CXX=<toolchain-prefix>/bin/x86_64-conda-linux-gnu-g++
export NVCC_PREPEND_FLAGS="-ccbin $CXX"
python setup.py build_ext --inplace
```

Set `TORCH_CUDA_ARCH_LIST` to the compute capability of the GPU you will run on
(`python -c "import torch; print(torch.cuda.get_device_capability())"` prints it
as a pair, so `(8, 6)` becomes `8.6`); listing only that architecture keeps the
build short. Point `CUDA_HOME` at the CUDA toolkit whose version matches
`torch.version.cuda`, and `<toolchain-prefix>` at the GCC installation from the
previous step.

`NVCC_PREPEND_FLAGS` is easy to miss: `CC`/`CXX` steer the `.cpp` files, but
`nvcc` compiles the `.cu` files with its own default host compiler
(`/usr/bin/c++`), and the C++20 check fails again partway through the build.

### Verify before installing

An extension that imports can still be numerically wrong. Compare it with the
reference implementation shipped beside it, on gradients as well as outputs:

```python
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
y_k = selective_scan_fn(u, dt, A, B, C, D, z=z, delta_bias=db, delta_softplus=True)
y_r = selective_scan_ref(u, dt, A, B, C, D, z=z, delta_bias=db, delta_softplus=True)
# forward and grads w.r.t. u, dt, B, C should agree to ~1e-7 in fp32
```

Back up the old `.so`, copy the new one into `site-packages` only once the
comparison passes, and re-run `test/streaming/test_dpcrn_streaming.py`. The
streaming path exercises `MambaInter.step()` rather than the scan, so it
confirms the exported graph is unchanged.
