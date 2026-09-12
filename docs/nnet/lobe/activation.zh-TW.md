# puresound.nnet.lobe.activation

English version: `activation.md`

依名稱建構 activation module 的工廠函式。

## Functions

### `get_activation(name: str) -> nn.Module`

回傳對應 `name` 的 PyTorch activation module **class**(不是 instance)。

**Parameters:**
- `name` – activation 名稱,需與下表完全相符(module 本身不會做小寫轉換;呼叫端如 `Unet` 會自行 `get_activation(activation_type.lower())`)

**Supported values:**

| Name | Module |
|------|--------|
| `"relu"` | `nn.ReLU` |
| `"mish"` | `nn.Mish` |
| `"prelu"` | `nn.PReLU` |
| `"sigmoid"` | `nn.Sigmoid` |
| `"tanh"` | `nn.Tanh` |

**Returns:** activation module class(需自行呼叫 `()` 來實例化)。

**Raises:** 若 `name` 不在上表中,拋出 `NameError`。實際實作如下:

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

下方兩個 `ValueError` 分支在目前程式碼中永遠不會被觸發:只要 `name`
沒有通過上面的白名單檢查——不論是不是字串——都會先拋出 `NameError`,因此
`cls` 不可能是 `None`,`else` 分支也永遠進不去。

## Wiring

`Unet` / `UnetTcn`(`puresound/nnet/unet.py`)會呼叫
`get_activation(activation_type.lower())`,依設定字串決定 encoder/decoder
conv block 內使用的非線性函式。

## Example

```python
from puresound.nnet.lobe.activation import get_activation

ActClass = get_activation("prelu")
act = ActClass()           # nn.PReLU()
out = act(features)
```
