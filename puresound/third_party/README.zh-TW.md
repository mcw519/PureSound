# 第三方程式碼（Third-Party Code）

此目錄存放 PureSound 選用功能所依賴的第三方研究程式碼（vendored code）。

## pytARD

`pytARD/` 取自 `https://github.com/gpuard/pytARD`。

- 用途：基於波動方程式（wave-based）的低頻房間脈衝響應（RIR）模擬。
- 授權：AGPL-3.0，詳見 `pytARD/LICENSE`。
- 整合點：`puresound.audio.rir.render.low_frequency.pytard.GpuARDPytARDBackend`
  （也從 `puresound.audio.rir.api` 重新匯出）。

上游專案以腳本形式組織，並未包裝成一般的 pip 相依套件。PureSound 透過一層
wrapper 匯入它，而不直接修改上游檔案。
