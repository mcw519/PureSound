# puresound.nnet.lobe.activation

English version: [activation.md](activation.md)

依名稱查出 activation module 的 class，讓 backbone config 可以用一個字串選擇非線性函式。

## Function: `get_activation`

```python
get_activation(name: str) -> type[nn.Module]
```

回傳的是 activation 的 **class**，不是 instance。

| `name` | class |
| --- | --- |
| `"relu"` | `nn.ReLU` |
| `"prelu"` | `nn.PReLU` |
| `"mish"` | `nn.Mish` |
| `"sigmoid"` | `nn.Sigmoid` |
| `"tanh"` | `nn.Tanh` |

比對是完全一致且區分大小寫；其他名稱一律拋出 `NameError`。呼叫端要先轉小寫。

## 在 recipe 中的用法

`Unet` 及其子類別（`UnetTcn`、`UnetFsmn`、`DPCRN`）會呼叫
`get_activation(activation_type.lower())`，所以 config 裡的值不分大小寫：

```yaml
backbone:
  type: DPCRN
  backbone_args:
    activation_type: PReLU
```

## 設計說明

回傳 class 讓 chassis 能為每一層 conv 各建一個新的 instance。這對有可學斜率的
`nn.PReLU` 很重要：若共用同一個 instance，所有層的斜率會被綁在一起。
