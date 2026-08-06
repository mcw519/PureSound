# puresound.nnet.lobe.activation

繁體中文版本：`activation.zh-TW.md`

Activation function factory for constructing activation modules by name.

## Functions

### `get_activation(name: str) -> nn.Module`

Returns a PyTorch activation module **class** (not an instance) for the given name.

**Parameters:**
- `name` – activation function name, matched exactly against the table below (the module does not itself lowercase `name`; callers such as `Unet` do `get_activation(activation_type.lower())`)

**Supported values:**

| Name | Module |
|------|--------|
| `"relu"` | `nn.ReLU` |
| `"mish"` | `nn.Mish` |
| `"prelu"` | `nn.PReLU` |
| `"sigmoid"` | `nn.Sigmoid` |
| `"tanh"` | `nn.Tanh` |

**Returns:** the activation module class (call it with `()` to instantiate).

**Raises:** `NameError` for any `name` not in the table above. Actual implementation:

```python
def get_activation(name: str) -> nn.Module:
    if name not in ["relu", "mish", "prelu", "sigmoid", "tanh"]:
        raise NameError("Could not interpret activation identifier")

    if isinstance(name, str):
        cls = globals().get(name)
        if cls is None:
            raise ValueError("Could not interpret activation identifier: " + str(name))
        return cls
    else:
        raise ValueError("Could not interpret activation identifier: " + str(name))
```

The two `ValueError` branches are unreachable in the current code: the whitelist
membership check above them raises `NameError` first for anything — string or
not — that isn't one of the 5 supported names, so `cls` can never end up `None`
and the `else` branch can never trigger.

## Wiring

`Unet` / `UnetTcn` (`puresound/nnet/unet.py`) call
`get_activation(activation_type.lower())` to pick the nonlinearity used inside
the encoder/decoder conv blocks from a config string.

## Example

```python
from puresound.nnet.lobe.activation import get_activation

ActClass = get_activation("prelu")
act = ActClass()           # nn.PReLU()
out = act(features)
```
