# 直達／早期／晚期歸因 — `puresound.audio.rir.metrics.attribution`

English version: [rir_attribution.md](rir_attribution.md)

把渲染出的響應拆成直達、早期反射與晚期反射三個分量，三者相加嚴格還原原始
響應。它用來稽核 renderer 如何在這三部分之間分配能量，不是感知指標。Schema
字串：`ATTRIBUTION_SCHEMA_VERSION = "puresound.rir_attribution.v1"`。

## 分解

```python
decompose_direct_early_later(
    full_output,            # 1-D 響應
    direct_anchor_output,   # 1-D 直達路徑分量，長度相同
    *, sample_rate_hz, split_center_s, transition_width_s,
) -> dict  # direct, early_reflections, later_reflections, early_cumulative, full
```

```text
r      = full − direct
early  = r · m_early(t)
later  = r · m_later(t),        m_later = 1 − m_early
early_cumulative = direct + early
```

**直達分量由呼叫端提供，而不是用時間窗切。** 當聲源脈衝比直達路徑到第一個反射
的延遲還寬時，固定時間窗會失效：直達脈衝的一部分會被算成反射。所以呼叫端單獨
渲染直達路徑——對 PathEvent 而言就是那唯一的 `direct` event
（`partition_path_events_by_arrival` 依到達時間把 event 分成直達、早期與晚期）——
並以 `direct_anchor_output` 傳入。

**分解在構造上就是精確的。** 兩個 mask 逐 sample 互補，所以不論 crossfade 形狀
為何，`direct + early + later == full`，誤差只到浮點捨入。

## Mask

```python
complementary_early_late_masks(
    *, num_samples, sample_rate_hz, split_center_s, transition_width_s,
) -> (early, later)
```

`early` 在 `split_center_s − transition_width_s / 2` 之前為 1，在過渡區內以
raised cosine `0.5 + 0.5 cos(π p)` 下降（`p` 由 0 到 1），之後為 0；
`later = 1 − early`。

## 檢查還原

```python
reconstruction_error(components) -> {"maximum_absolute_error": float, "nrmse": float}
```

需要 `direct`、`early_reflections`、`later_reflections` 與 `full`（正是
`decompose_direct_early_later` 的回傳），把三部分相加，回報最大絕對誤差與誤差
範數除以 `‖full‖₂`。它把「精確還原」這個主張變成驗證器可以斷言的數字。

## 使用處

PathEvent renderer 的驗證用它稽核相干高頻段的直達／早期／晚期拆分；
`test/rir/test_rir_attribution.py` 固定住還原性質。
