# puresound.utils

繁體中文版本：[utils.zh-TW.md](utils.zh-TW.md)

General utility functions for file I/O, configuration loading, and tensor
operations.

## Functions

### `str2bool(v: str) -> bool`

```python
def str2bool(v: str):
    return v.lower() in ("true", "yes")
```

Only the (case-insensitive) strings `"true"` and `"yes"` return `True`.
**Everything else — `"1"`, `"t"`, `"y"`, `"false"`, `"no"`, `"0"`, typos, an
empty string — returns `False`.** It never raises; there is no validation
of the input at all.

---

### `str2list(s: str) -> List`

Splits a whitespace-delimited string into a Python list (`s.strip().split()`
— any run of whitespace is a separator, not only a single space).

---

### `load_text_as_dict(file_path: str, separator: str = " ", coding: str = "utf8") -> Dict[str, List[str]]`

Loads a delimited text file into a dict. The first column of each line
becomes the key; **the remaining columns always become a list of strings**
— even when there's exactly one remaining column, the value is a
one-element list, not a bare string (`{"aaa": ["bbb"]}`, not `{"aaa":
"bbb"}`).

**Parameters:**
- `file_path` – path to the text file
- `separator` – column delimiter (default `" "`)
- `coding` – file encoding (default `"utf8"`)

---

### `recursive_read_folder(folder: str, file_type: str, output: Optional[List]) -> None`

```python
def recursive_read_folder(folder: str, file_type: str, output: Optional[List]) -> None:
```

**`output` has no default** despite the `Optional[List]` hint — it is a
required positional/keyword argument. Pass in an existing list; the
function **mutates it in place and returns `None`**:

```python
found = []
recursive_read_folder("corpus", ".flac", found)
# found is now populated; the function's own return value is None
```

Each entry is a string `"<file name> <full path>"` (an scp-style line), not a
bare path.

Passing `output=None` (or omitting it) raises as soon as a matching file is
found (`None.append(...)`).

Matching is a plain **substring test** (`file_type in file`), not a suffix
check — `file_type=".wav"` also matches a filename like
`"backup.wav.old"`.

`iter_files_recursive(folder, file_type)` is the same walk (depth-first, in
`os.listdir` order, same substring match) as a generator of
`(file name, full path)` pairs. Use it when a name or directory may contain a
space, which the scp-style line cannot split back apart.

---

### `load_hparam(file_path: str) -> Dict`

Loads a YAML file via `yaml.safe_load_all` — so it supports multiple
`---`-separated documents in one file, merging every document's top-level keys
into a single flat dict, in order (a later document's key overwrites an earlier
document's same key). An empty document is skipped. The file is read as UTF-8
and closed on return. Python-specific tags (`!!python/...`) are refused with a
`yaml.YAMLError`.

---

### `pin_thread_pools() -> None`

Pins every thread pool the calling process could spawn to one thread: torch's
intra-op pool (`set_num_threads`) and inter-op pool (`set_num_interop_threads`,
skipped when the process has already run a parallel region), numba, and the
BLAS/OpenMP pools through `threadpoolctl`; it also sets `OMP_NUM_THREADS` for
libraries imported later. Meant for worker processes, which are forked with the
parent's thread counts already decided and would otherwise each run one thread
per core. The training DataLoader workers (`system.runner.seed_worker`) and the
scoring pool (`evaluation.parallel`) both call it. numba and threadpoolctl are
optional; a missing one is skipped.

---

### `create_folder(folder_name: str) -> None`

Creates a directory (and any intermediates) if it doesn't already exist, via
`os.makedirs(folder_name, exist_ok=True)`. It never raises `FileExistsError`: a
collision with a concurrent creator (several DataLoader workers or DDP ranks
creating the same folder) is logged at debug level and ignored.

---

### `convolve(x: torch.Tensor, filter: torch.Tensor) -> torch.Tensor`

Direct time-domain 1-D convolution, left-zero-padded by `len(filter) - 1`
samples so the output is causal and the same length as the input.

**Shapes**: `x` is `[1, T]` (single channel), `filter` is `[K]` (a 1-D
kernel); returns `[1, T]`.

---

### `next_fast_len(size: int) -> int`

Returns the next integer ≥ `size` whose only prime factors are 2, 3, and 5
(an efficient FFT size) — equivalent to `scipy.fftpack.next_fast_len`.
Results are memoized in a module-level cache (`_NEXT_FAST_LEN`), keyed by
the requested `size`.

---

### `fftconvolve(x: torch.Tensor, kernel: torch.Tensor, mode: str = "full") -> torch.Tensor`

FFT-based convolution (via `torch.fft.rfft`/`irfft`, internally rounded up
to a fast FFT length).

- `"full"` – length `len(x) + len(kernel) - 1`
- `"same"` – length `max(len(x), len(kernel))`, centred in the full result —
  the length of the **longer** input, not always `len(x)`, if `kernel` happens
  to be the longer one
- `"valid"` – length `max(len(x), len(kernel)) - min(len(x), len(kernel)) +
  1` (only the region where both signals fully overlap)

Convolving a waveform with an RIR in `"full"` mode introduces a propagation
delay whenever the RIR's peak isn't at sample 0 — locate it with
`rir.abs().argmax(dim=-1)` and trim accordingly.
