# 第三方程式碼（Third-Party Code）

PureSound 可以使用、但**不隨套件散布**的選用第三方相依。此目錄不包含任何
第三方程式碼；以下說明各項要自行安裝什麼、以及 PureSound 如何找到它。

English version: [`README.md`](README.md)

## pytARD —— 選用，使用者自行安裝

基於波動方程式的低頻房間脈衝響應模擬，由
`puresound.audio.rir.render.low_frequency.pytard.GpuARDPytARDBackend` 使用
（也從 `puresound.audio.rir.api` 重新匯出）。

- 上游：<https://github.com/gpuard/pytARD>
- 授權：**AGPL-3.0**

本專案不轉散布它，而它也不在 PyPI 上，因此無法做成 pip extra。請自行安裝：

```bash
git clone https://github.com/gpuard/pytARD.git /path/to/pytARD
export PURESOUND_PYTARD_ROOT=/path/to/pytARD
```

PureSound 依下列順序尋找 checkout：

1. 傳給 backend 的 `third_party_root` 參數，
2. `$PURESOUND_PYTARD_ROOT`，
3. `puresound/third_party/pytARD`——所以自行 vendored 的安裝方式仍然有效。

Backend 在模擬時才從 checkout 根目錄匯入 `pytARD_3D` 與 `common`，且不修改
上游檔案。沒有 checkout 時模組仍可匯入，只有這個 backend 會丟錯，驅動 solver
的測試會 skip。

**其他低頻 backend 完全不受影響**——`AnalyticModalLowFrequencyBackend` 與
`ImpedanceModalLowFrequencyBackend` 是 PureSound 自己的程式碼，不需安裝任何東西。
