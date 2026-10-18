# puresound.nnet.lobe.onnx_module

English version: [onnx_module.md](onnx_module.md)

把小型 ONNX 圖當成可微分的 torch module 執行。公開的裁判（DNSMOS 這類品質預測器、VAD）
通常以 ONNX 檔發佈；onnxruntime 能算分但不給梯度。用 torch op 執行同一張圖，它就能當
loss 項：每個權重都是凍結的 buffer，輸入保留 autograd 歷史。

## Class: `OnnxModule`

```python
OnnxModule(path: str | Path)
forward(*inputs: Tensor) -> Tensor | list[Tensor]
```

`input_names` / `output_names` 取自圖；initializer 變成 non-persistent buffer、`requires_grad` 關閉，
所以這個 module 沒有參數，也不會被存進 checkpoint。節點依圖的順序執行，因此形狀來自
initializer 的 `Reshape`、邊界來自 initializer 的 `Slice` 會把它們當常數讀。

支援的 op：`Conv`、`MaxPool`、`AveragePool`、`GlobalMaxPool`、`GlobalAveragePool`、
`MatMul`、`Gemm`、`Add`、`Sub`、`Mul`、`Div`、`Sqrt`、`Pow`、`Log`、`Exp`、`Max`、`Min`、
`Clip`、`Relu`、`Sigmoid`、`Tanh`、`ReduceMax`、`ReduceMean`、`Slice`、`Reshape`、`Flatten`、
`Concat`、`Transpose`、`Unsqueeze`、`Squeeze`、`Identity`。含其他 op 的圖在載入時就
`ValueError` 並指名該 op，不會等到第一次 forward。

`Sqrt` 與 `Log` 的輸入以 `GRAD_FLOOR = 1e-12` 做下限：它們在 0 的導數無窮大，
一個恰好為零的 STFT bin（靜音幀）就會把整批的梯度變成 NaN。這個下限遠低於裁判會看到的任何數值，
forward 不變。

## 用途

[`DNSMOSLoss`](../../losses/dnsmos.zh-TW.md) 用它包 DNSMOS P.835 裁判。這個 module 不懂音訊：
它只是那張圖。
