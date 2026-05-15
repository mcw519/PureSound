# puresound.nnet.lobe.activation

Activation function factory for constructing activation modules by name.

## Functions

### `get_activation(name: str) -> nn.Module`

Returns a PyTorch activation module class (not instance) for the given name.

**Parameters:**
- `name` – Activation function name (case-insensitive)

**Supported values:**

| Name | Module |
|------|--------|
| `"relu"` | `nn.ReLU` |
| `"mish"` | `nn.Mish` |
| `"prelu"` | `nn.PReLU` |
| `"sigmoid"` | `nn.Sigmoid` |
| `"tanh"` | `nn.Tanh` |

**Returns:** The activation module class (call it with `()` to instantiate).

**Raises:** `ValueError` for unrecognized activation names.

## Example

```python
from puresound.nnet.lobe.activation import get_activation

ActClass = get_activation("prelu")
act = ActClass()           # nn.PReLU()
out = act(features)
```
