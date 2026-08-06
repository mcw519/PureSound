# Direct/early/later 歸因（attribution）— `puresound.audio.rir.metrics.attribution`

English version: `rir_attribution.md`

把一段 render 出來的 response，拆成 direct（直達音）、early reflections（早期
反射）、later reflections（晚期反射）三個分量，且三者相加會精確等於原始
response——這是用來稽核 renderer 自身內部能量分配是否合理的工具，**不是**
一個聽感（perceptual）指標。Policy 字串：
`ATTRIBUTION_SCHEMA_VERSION = "puresound.rir_attribution.v1"`。

## 為什麼 direct 分量是外部提供，不是用時間窗切出來的

`decompose_direct_early_later(full_output, direct_anchor_output, *,
sample_rate_hz, split_center_s, transition_width_s)` 把 direct-path 分量當成
「外部提供的第二個陣列」，而不是用一個固定時間窗從 `full_output` 裡推斷出來。
用固定時間窗會在「音源脈衝本身的寬度」大於「direct 到第一個反射之間的延遲」
時失效——這種時候窗會把 direct 脈衝的一部分切進本該屬於 early reflection 的
範圍。改成外部提供 `direct_anchor_output`（通常是同一個音源、走同一條路徑，
在自由場／anechoic 條件下的 render 結果）就完全避開了這個失效模式。

殘差 `full - direct` 再由一組互補的 early/later mask 拆開，因此這個分解
**依構造即為精確可重建（exactly reconstructive by construction）**——
`direct + early_reflections + later_reflections == full`，除了浮點數捨入
誤差外完全成立，不是「大致」成立。

回傳一個 dict：`direct`、`early_reflections`、`later_reflections`、
`early_cumulative`（`direct + early_reflections`）、`full`。

## 互補遮罩（complementary mask）

```python
complementary_early_late_masks(
    *, num_samples, sample_rate_hz, split_center_s, transition_width_s,
) -> (early: np.ndarray, later: np.ndarray)
```

一個 raised-cosine 的 crossfade：`early` 在 transition 窗之前是 `1.0`，接著在
以 `split_center_s` 為中心、寬度 `transition_width_s` 的窗內平滑降到 `0.0`，
之後維持 `0.0`。依構造 `later = 1.0 - early`，因此逐 sample 都滿足
`early + later == 1.0`——真正讓這個分解精確成立的是**這個恆等式**，而不是
crossfade 本身的形狀。

## 驗證重建結果

```python
reconstruction_error(components: Mapping[str, np.ndarray]) -> dict
# {"maximum_absolute_error": float, "nrmse": float}
```

要求 `components` 內含 `direct`、`early_reflections`、`later_reflections`、
`full`（正好就是 `decompose_direct_early_later` 回傳的內容）；把三個分量加
總後，回報相對 `full` 的絕對誤差與（以 `full` 的 L2 範數正規化的）nrmse。
用來從數值上驗證「精確可重建」這個宣稱，而不只是依構造相信它。

## 接線（wiring）

由
`egs/rir_generation/phases/m3_wave_path/scripts/validate_direct_early_later_attribution.py`
用來稽核 M3 PathEvent renderer 自身的 direct/early/later 拆分——M3 backend
如何產出這個模組所消費的 `direct_anchor_output`，見
[`rir_realism_algorithm.zh-TW.md`](rir_realism_algorithm.zh-TW.md) §4.2。
