# puresound.utils

General utility functions for file I/O, configuration loading, and tensor operations.

## Functions

### `str2bool(v: str) -> bool`

Converts a string representation to a boolean value.

- `"yes"`, `"true"`, `"t"`, `"y"`, `"1"` → `True`
- `"no"`, `"false"`, `"f"`, `"n"`, `"0"` → `False`

Raises `argparse.ArgumentTypeError` for unrecognized values.

---

### `str2list(s: str) -> List`

Splits a whitespace-delimited string into a Python list.

**Parameters:**
- `s` – Input string

**Returns:** List of string tokens.

---

### `load_text_as_dict(file_path: str, separator: str = " ", coding: str = "utf8") -> Dict`

Loads a tab- or space-separated text file into a dictionary. The first column becomes the key; remaining columns form the value.

**Parameters:**
- `file_path` – Path to the text file
- `separator` – Column delimiter (default: `" "`)
- `coding` – File encoding (default: `"utf8"`)

**Returns:** `Dict[str, str]`

---

### `recursive_read_folder(folder: str, file_type: str, output: Optional[List] = None)`

Recursively discovers all files with the given extension under a directory.

**Parameters:**
- `folder` – Root directory to search
- `file_type` – File extension (e.g., `".wav"`)
- `output` – Existing list to append results to (optional)

**Returns:** List of absolute file paths.

---

### `load_hparam(file_path: str) -> Dict`

Loads a YAML configuration file into a Python dictionary.

**Parameters:**
- `file_path` – Path to the YAML file

**Returns:** `Dict` of hyperparameters.

---

### `create_folder(folder_name: str)`

Creates a directory (and any intermediate directories) if it does not already exist.

**Parameters:**
- `folder_name` – Path to the directory to create

---

### `convolve(x: torch.Tensor, filter: torch.Tensor) -> torch.Tensor`

Performs 1D convolution on two tensors using direct time-domain convolution.

**Parameters:**
- `x` – Input signal tensor
- `filter` – Convolution kernel tensor

**Returns:** Convolved output tensor.

---

### `next_fast_len(size: int) -> int`

Returns the next integer ≥ `size` whose prime factors are only 2, 3, and 5 (an efficient FFT size).

**Parameters:**
- `size` – Minimum required size

**Returns:** Optimal FFT length.

---

### `fftconvolve(x: torch.Tensor, kernel: torch.Tensor, mode: str = "full") -> torch.Tensor`

FFT-based convolution, faster than direct convolution for long signals.

**Parameters:**
- `x` – Input signal tensor
- `kernel` – Convolution kernel tensor
- `mode` – Output mode:
  - `"full"` – Full convolution output (length `N + M - 1`)
  - `"same"` – Output same length as `x`
  - `"valid"` – Only values where both signals fully overlap

**Returns:** Convolved output tensor.

## Constants

| Name | Description |
|------|-------------|
| `_NEXT_FAST_LEN` | Internal cache mapping sizes to their next fast FFT length |
