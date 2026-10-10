# puresound.nnet.lobe.activation

繁體中文版本：[activation.zh-TW.md](activation.zh-TW.md)

Looks up an activation module class by name, so a backbone config can choose
its nonlinearity with a string.

## Function: `get_activation`

```python
get_activation(name: str) -> type[nn.Module]
```

Returns the activation **class**, not an instance.

| `name` | class |
| --- | --- |
| `"relu"` | `nn.ReLU` |
| `"prelu"` | `nn.PReLU` |
| `"mish"` | `nn.Mish` |
| `"sigmoid"` | `nn.Sigmoid` |
| `"tanh"` | `nn.Tanh` |

The match is exact and case-sensitive; any other name raises `NameError`.
Callers lowercase first.

## Use in a recipe

`Unet` and its subclasses (`UnetTcn`, `UnetFsmn`, `DPCRN`) call
`get_activation(activation_type.lower())`, so the config value is
case-insensitive:

```yaml
backbone:
  type: DPCRN
  backbone_args:
    activation_type: PReLU
```

## Design notes

Returning the class lets the chassis build a fresh instance for every conv
layer. This matters for `nn.PReLU`, which has a learned slope: one shared
instance would tie the slope across all layers.
