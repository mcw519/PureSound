# puresound.nnet.loss

English version: [index.md](index.md)

Loss 函式庫。Recipe 在 `loss_func` 底下列出要用的 loss；每一項的 `type` 是
class 名稱，由 `puresound.recipes.init_loss_func` 以 `getattr` 在
`puresound.nnet.loss` 上解析，`args` 則是它的 constructor 參數。Class 必須列在
`puresound.nnet.loss.__all__` 裡才取得到。

```yaml
loss_func:
  - type: SDRLoss
    weighted: 1.0
    args: {...}
```

Loss 以 `required_inputs` 宣告自己要被餵什麼：一個 provider 名稱的 tuple，順序
與 `forward` 的參數一致；訓練模組（`puresound.system.base.invoke_loss`）逐一到
自己的 provider 表查這些名稱，缺一個就直接報錯。沒有宣告的 loss 會拿到
`("enhanced", "target")`。Provider 包括波形、batch dict、VAD target，以及
backbone 的 side output（`vad_logits`、`dist_preds`、`identity_emb`、
`proximity` 等），這些只有在 backbone 開啟對應 head 時才存在。

| 頁面 | Classes | 狀態 | 計算內容 |
| --- | --- | --- | --- |
| [sdr](sdr.zh-TW.md) | `SDRLoss` | active | 時域 SDR 系列（SI-SNR、SD-SDR、SA-SDR、t-SDR 等），target-absent rows 另走專門路徑 |
| [stft_loss](stft_loss.zh-TW.md) | `MultiResolutionSTFTLoss`、`SpectralLoss`、`OverSuppressionLoss` | active | 頻譜距離；`OverSuppressionLoss` 是只懲罰被去掉之 target 能量的單邊項 |
| [active_bins](active_bins.zh-TW.md) | `ActiveBinLogMagLoss` | active | 逐 bin 的 log 幅度誤差，只算在 clean target 有語音的 bin 上 |
| [asr_feature](asr_feature.zh-TW.md) | `ASRFeatureLoss` | active | 透過凍結的自監督語音模型做 feature matching，作為字錯誤的可微分 proxy |
| [residual](residual.zh-TW.md) | `ResidualReferenceLoss` | active | 把 `noisy - enhanced` 監督到參考殘差上 |
| [dist](dist.zh-TW.md) | `DistHeadRegressionLoss` | active | 給 `DistHead` 用、NaN-masked 的距離 / DRR regression |
| [vad](vad.zh-TW.md) | `VADActivityLoss`、`VADHeadBCELoss`、`BackgroundVADHeadBCELoss`、`F1_loss` | active | 對增強輸出與 VAD head 的逐 frame 活動監督 |
| [identity](identity.zh-TW.md) | `IdentityContrastiveLoss` | library | 在 `IdentityHead` 上、對 EMA teacher 做 turn 層級的語者對比 |
| [proximity](proximity.zh-TW.md) | `RelativeProximityLoss` | active | 依渲染距離，在 `ProximityHead` 讀出值上排序使用者與旁人的 turn |
| [inherit](inherit.zh-TW.md) | `AnchorInheritanceLoss` | library | hinge：使用者起音處的增益須高於對其之前內容所施加的增益 |
| [spk](spk.zh-TW.md) | `AAMsoftmax`、`SphereFace2`、`GE2ELoss`、`TripletLoss` | 凍結的 legacy | speaker-verification 與 TSE recipe 用的語者 embedding loss |
| —（`loss/__init__.py`） | `TimeDomainBasicLoss` | library | 波形間的單純 L1 或 MSE（`name: l1 \| mse`、`reduction`） |

- **active**——有維護中的 recipe 在用（module 內至少一個 class）。
- **library**——可 import、有測試，但沒有維護中的 recipe 使用。
- **凍結的 legacy**——為既有 recipe 保留，不再開發。
