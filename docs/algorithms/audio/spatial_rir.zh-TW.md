# 空間 RIR 與 BRIR — `puresound.audio.rir.render.spatial`

English version: [spatial_rir.md](spatial_rir.md)

把一個聲源渲染到一組**同步**的 receiver——麥克風陣列、一階 Ambisonics（FOA），
以及可選的雙耳對——讓 channel 間的延遲、相干性與 IACC 都來自同一個物理聲場。
它是獨立、需明確選用的 API：bank 生成（`generate_hybrid_rir`）每個 item 只渲染
一個單聲道 receiver，不使用它。

## 入口

```python
from puresound.audio.rir.render.spatial import render_room_scene_spatial_rir

render = render_room_scene_spatial_rir(
    scene,                          # 有一個或多個 receiver 的 RoomSceneV2
    sample_rate=16000, duration_s=1.2, source_index=0,
    max_order=4,                    # PathEvent image-source 階數（0..20）
    mixing_time_s=0.024,            # 每個直達到達之後的 transition 中心
    transition_duration_s=0.016,
    delay_line_count=16,            # FDN delay line 數（2 的冪次）
    plane_wave_count=128,           # 晚場方向數（>= 16）
    seed=...,                       # 晚場的基礎 seed
    minimum_fdn_center_hz=500.0,
    material_reference_frequency_hz=1000.0,
    decoder=None,                   # 產生 BRIR 用的 AmbisonicBinauralDecoder
)
```

`SpatialRoomRIRRender` 包含 `receiver_rirs` `[receivers, samples]`、
`ambisonic_acn_sn3d` `[4, samples]`（ACN 順序 `W, Y, Z, X`，SN3D 正規化）、兩者
各自的純相干與純晚場分量、耦合紀錄、可選的 `binaural` BRIR
`[2, samples + taps − 1]`，以及 metadata（policy `puresound.spatial_room_rir.v1`）。

## 演算法

1. **早場，逐 receiver。** 為每個 receiver 產生相干 PathEvent（以材質 relaxation
   先驗建邊界濾波、指向性、物件遮擋），並依該 receiver 的直達距離套用
   ISO 9613-1 空氣吸收濾波。FOA 的早場使用位於陣列質心的全指向參考 receiver。
2. **一個共用晚場。** 以 scene 在 500 Hz – 4 kHz 的預測 octave RT60（取
   `minimum_fdn_center_hz` 以上且低於 Nyquist 者）並經空氣吸收修正，設計一個
   multiband FDN（[FDN](multiband_fdn.zh-TW.md)）。它由最早直達到達時刻的單位脈衝
   激發。`plane_wave_count` 個方向取自依 seed 旋轉的 Fibonacci 球面
   （`fibonacci_sphere_directions`）。每個 octave 頻帶中，每個方向帶一個獨立的
   高斯 octave 頻帶噪音載波，乘上該 FDN 頻帶的因果 RMS 包絡（10 ms）；總和再縮放
   到讓 W channel 帶有 FDN 的能量。
3. **投影。** 每個 receiver 把所有平面波相加，每道波相對陣列質心 `r̄` 延遲
   `(r − r̄) · k / c`（再加一個等於孔徑半徑除以 `c` 的共同因果餘量），並乘上該
   receiver 對此到達方向的指向性。FOA 是在質心的投影：`W = Σ p`、`Y = Σ n_y p`、
   `Z = Σ n_z p`、`X = Σ n_x p`，以 `√N` 正規化。
4. **早晚場耦合。** 每個 channel 套用[晚場耦合](rir_late_coupling.zh-TW.md)的等功率
   crossfade，中心在它自己的直達到達後 `mixing_time_s`，所以 transition 之前的
   sample 完全不變。接著所有 channel 的擴散分量乘上**同一個共用增益**：讓陣列在
   transition 之後的總能量等於各 channel 目標（各自由材質 RT60 外推）總和的正根。
   共用增益保住 channel 間的能量比例——也就是空間線索——逐 channel 重新正規化會把
   它抹掉（`"energy_policy": "one_shared_array_gain_preserves_spatial_ratios"`）。
5. **雙耳（可選）。** `render_ambisonic_brir(foa, sample_rate, decoder)` 把四個 FOA
   channel 與 decoder 的因果 FIR 卷積後逐耳相加。

由於每個 channel 都是同一組由單一 FDN 驅動之平面波的投影，晚場具有物理上一致的
channel 間相干性，而不是各自獨立生成的單聲道尾巴。

## 指向性

聲源與 receiver 使用實數一階聲壓指向型
（`puresound.audio.rir.path_events` 的 `directivity_pressure_gain`）：

```text
gain = α + (1 − α) · (forward · direction)
```

| 指向型 | α |
|---|---|
| `omnidirectional` | 1（增益為 1，與朝向無關） |
| `cardioid`、`speech_cardioid` | 0.5 |
| `hypercardioid` | 0.25 |
| `figure_eight` | 0 |

`forward` 由 pose 的 yaw 與 pitch 決定。Hypercardioid 與 figure-eight 在後瓣為負；
這個正負號是真實的聲壓相位反轉，不會被截成零。未知的指向型會 raise
`NotImplementedError`，而不是退回全指向。

## 雙耳 decoder

`AmbisonicBinauralDecoder`（`puresound.audio.rir.render.binaural`）是形狀為
`[2 耳, 4 FOA channel, taps]` 的因果雙耳 FIR 組。建構時要求正的取樣率、上述形狀、
有限的 tap、`reference_id` 與 `decoder_kind`，並保存一個 `provenance` mapping；
`render_ambisonic_brir` 要求 FOA 取樣率與 decoder 相同。

`analytic_first_order_binaural_decoder(sample_rate)` 是單 tap、側向、無頭部模型
的 decoder（`decoder_kind="analytic_demonstration_not_hrtf"`），只用來走通流程；
其 provenance 記錄 `"measurement": false` 與
`"production_hrtf_replacement_required": true`。真正的雙耳渲染透過同一個類別
注入量測所得、具授權的 HRTF 衍生 FIR；空間 renderer 不需要改變。

## 命令列

```bash
PYTHONPATH=. python egs/rir_generation/render_spatial_rir.py \
  --sample-rate 16000 --duration 1.2 \
  --binaural-spacing-m 0.17 --binaural-decoder analytic \
  --output-dir <out>
```

沒有 `--scene-json` 時工具會抽一個決定性的辦公室 scene；`--binaural-spacing-m`
把單一 receiver 的 scene 展開成沿 x 軸的一對（`<= 0` 則停用）。`--max-order`、
`--delay-lines`、`--plane-waves`、`--source-index` 與 `--seed` 對應上面的參數。

`puresound.audio.rir.render.arrays` 中的陣列工具（`pad_or_trim`、
`coerce_rir_array`）與 receiver 陣列無關：它們只負責 backend 與 crossover 共用的
`[channels, samples]` float64 排版。
