# M4 空間 RIR 與 BRIR

English version: `spatial_rir.md`

M4 的空間 renderer 是明確 opt-in 的 API；既有 dataset generator 仍預設
Pyroomacoustics。高階入口是
`puresound.audio.rir.render.spatial.render_room_scene_spatial_rir()`，輸入
`RoomSceneV2`，一次回傳：

- 同一 source 到所有同步 receivers 的完整 RIR；
- ACN/SN3D first-order Ambisonics，channel 固定為 `W/Y/Z/X`；
- coherent PathEvent early 與 shared spatial-FDN late components；
- 若提供 `AmbisonicBinauralDecoder`，再回傳兩聲道 BRIR。

late field 不是每個 channel 各抽亂數。所有 receivers 與 FOA 都投影自同一組
seeded Fibonacci-sphere 平面波，因此 inter-channel delay、coherence 與 IACC
有共同物理來源。direct/early 部分逐 receiver 使用 PathEvents；mixing
transition 前的 samples 完全保留，transition 後再以每 channel 的 positive
quadratic root 保存有限 RIR energy。

接收器支援 `omnidirectional`、`cardioid`、`hypercardioid` 與
`figure_eight` real pressure patterns。hypercardioid／figure-eight 背面的負
gain 是 pressure phase reversal，不會被 clamp。

BRIR decoder 的 FIR shape 是 `[2, 4, taps]`，並強制保存 sample rate、
reference ID、decoder kind 與 provenance。內附 analytic decoder 只供 pipeline
與 audition，明確不是 measured HRTF；production 應注入有授權與處理紀錄的
HRTF-derived FIR。

快速生成：

```bash
PYTHONPATH=. python egs/rir_generation/render_spatial_rir.py \
  --sample-rate 16000 --duration 1.2 \
  --binaural-spacing-m 0.17 --binaural-decoder analytic \
  --output-dir egs/rir_generation/exp/rir_realism/m4/spatial_demo
```

正式驗證：

```bash
PYTHONPATH=. python egs/rir_generation/phases/m4_spatial_late_field/scripts/validate_spatial_rir.py
PYTHONPATH=. python egs/rir_generation/phases/m4_spatial_late_field/scripts/validate_binaural_brir.py
PYTHONPATH=. python egs/rir_generation/phases/m4_spatial_late_field/scripts/validate_m4_exit.py
```

目前 M4 implementation exit 已通過；measured multi-receiver、licensed HRTF
calibration 與受控 listening 仍是獨立的 empirical／production exit。詳細公式
與正式數值見[繁中算法文件](rir_realism_algorithm.zh-TW.md)。
