# puresound.nnet.loss.dnsmos

English version: [dnsmos.md](dnsmos.md)

把 DNSMOS P.835（Reddy et al., 2022）當成可微分的訓練項。公開的
`sig_bak_ovr.onnx` 裁判透過 [`OnnxModule`](../models/lobe/onnx_module.zh-TW.md)
以 torch op 執行，增強器因此能從評測報 SIG / BAK / OVRL 的同一個網路拿到梯度。
不需要 clean 參考：真實錄音也能用。

## Class: `DNSMOSLoss`

### 計算內容

對增強輸出的每一列：

```
x      = 以不帶梯度的係數把列拉到 reference_rms（係數夾在 [1/max_scale, max_scale]）
raw    = judge(tile(x, 144160 樣本))                # 16 kHz 的 9.01 s；校準前的 sig、bak、ovr
MOS_k  = polyfit_k(raw_k)                           # 指標自己的校準多項式，所以數值是 MOS 單位
loss   = sum_k w_k * (5 - MOS_k) / 4                # 只算權重非零的分數
       + floor_weight * relu(sig_floor - MOS_sig)   # 只在設定 sig_floor 時
```

對有被評分的列取平均。RMS 低於 `silence_rms` 的列不送裁判（在圖內會 0 除以 0，
污染整批每一列的梯度）；整批都沒有可評分的列時回傳 `enh.sum() * 0.0`，
一個保留圖邊的零。短於 9.01 s 的列會鋪滿，較長的列取前 9.01 s。裁判以 fp32 執行、
關閉 autocast，不論上層模組的模式都維持 eval。

裁判把絕對位準也一起評（它的輸入是原始 log 功率頻譜）：在真實錄音上改 3 dB，
SIG 平均動 0.2–0.3。評分前先固定位準，這一項才不會獎勵「輸出變小聲或大聲」而非「變乾淨」。

### 建構子

```python
DNSMOSLoss(
    model_path: str = "~/.torchmetrics/DNSMOS/DNSMOS/sig_bak_ovr.onnx",  # torchmetrics 的下載位置
    sig_weight: float = 0.0,      # 各分數 (5 - MOS)/4 的權重
    bak_weight: float = 1.0,
    ovr_weight: float = 0.0,
    sig_floor: float | None = None,   # SIG 低於此值的 hinge（MOS 單位）
    floor_weight: float = 1.0,
    sample_rate: int = 16000,     # 其他取樣率直接報錯，這一項不重取樣
    max_rows: int | None = None,  # 每次只評隨機子集（每列約 0.45 GB activation）
    reference_rms: float | None = 0.05,  # −26 dBFS；None 以自然位準評分
    silence_rms: float = 1e-3,    # 比這安靜的列不評分
    max_scale: float = 20.0,      # 每列位準係數的上限
)
```

DNSMOS 檔案由 torchmetrics 在第一次跑任何 DNSMOS 指標時下載；找不到檔案會
`FileNotFoundError`。沒有任何權重也沒有 floor 的設定會 `ValueError`。

### 輸入

`forward(enh) -> Tensor`，波形 `[N, T]`（單一 channel 軸會被 squeeze）。
`required_inputs = ("enhanced",)`。

`scores(wav) -> (dict, mask)` 回傳校準後的 `sig` / `bak` / `ovr` 張量 `[N]`（帶梯度）
與有被評分之列的 `[N]` 遮罩；被排除的列讀到 0。

### Config 用法

無參考的裁判單獨用會被騙，所以這一項要放在參考式的項與學來的 critic 旁邊，權重適中：

```yaml
loss_func:
  - type: DNSMOSLoss
    weighted: 1.0
    args:
      sig_weight: 1.0
      bak_weight: 1.0
      max_rows: 6
```
