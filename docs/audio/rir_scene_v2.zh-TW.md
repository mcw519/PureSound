# RIR scene schema

English: [rir_scene_v2.md](rir_scene_v2.md)

`puresound.audio.rir.scene.schema` 定義 `rir_scene.v2`。Scene 保存幾何、
材料、環境與 transducers 等物理輸入；RT60 則由這些輸入推導。

## Schema

| Type | 用途 |
|---|---|
| `MaterialSpectrum` | 頻率值與 uncertainty，使用 log-frequency interpolation |
| `SurfaceMaterial` | Absorption、scattering、transmission、provenance 與可選 impedance |
| `SceneSurface` | 有名稱的房間邊界 mesh |
| `SurfacePatch` | 佔用部分邊界的 window 或 door material |
| `EnvironmentConfig` | 溫度、濕度、氣壓與音速 |
| `Pose` | Position 與 yaw/pitch/roll |
| `TransducerConfig` | Source/receiver identity、pose、pattern、level 與 calibration |
| `SceneObject` | 家具 footprint、高度、材料與 transmission |
| `RoomSceneV2` | 完整、經驗證且可序列化的 scene |

`to_json()` 與 `from_json()` 可 round-trip canonical scene。
`to_metadata()` 也會輸出既有 bank reader 所需的 compatibility fields。
`rt60` compatibility field 是 500/1000 Hz Sabine prediction 的中位數，不會
沿用 v1 requested RT60。

## Material catalog

`puresound.audio.rir.scene.materials` 包含常見室內表面的 population priors。
它們是 simulation priors，不是特定安裝產品的認證量測。

Room sampling 使用彼此相關的 room-level 與 material-level perturbations。
Room-type recipe 會選擇合理的牆面、地板、天花板、窗、門與家具組合。

Surface patches 會個別保留在 metadata。Shoebox renderer 使用依面積加權的
effective material。

## Complex impedance

Surface 可保存 Pa·s/m 單位的 real 與 imaginary impedance。兩個 spectra
必須同時存在、frequency centers 相同，而且 real impedance 不得為負。

Absorption 無法決定 impedance phase，因此 catalog 不會為 absorption-only
material 自動建立 impedance。

當 patched boundary 的所有 components 都有 impedance 時，effective
admittance 依面積混合：

```text
Y_effective = sum(area_fraction_i / Z_i)
Z_effective = 1 / Y_effective
```

任何 component 缺少 impedance 時，effective impedance 維持 unknown。
Absorption、scattering 與 transmission 仍可依面積加權。

詳見 [Impedance priors](impedance_priors.zh-TW.md) 與
[複數阻抗](impedance_measurements.zh-TW.md)。

## Rendering behavior

Pyroomacoustics 會取得每個 boundary 的 frequency-dependent absorption 與
scattering。

PathEvents 另外支援：

- explicit coherent reflection paths；
- source directivity；
- 有界的 object transmission 與 early-path occlusion；
- first-order reference diffraction 與 scattering。

PathEvents + FDN 保留 coherent early response，再加入 deterministic multiband
late field。

模型限制：shoebox backend 不把 surface patch 當成 explicit polygon；
diffraction 不是 general mesh solver；receiver pattern 也只在 backend 明確支援
時可用。

## Backend defaults

Default 屬於 CLI entry point，不屬於 scene schema：

| Entry point | Default |
|---|---|
| `egs/rir_generation/generate_hybrid_rir.py` | `pyroomacoustics` |
| `egs/rir_generation/phases/m6_bank/scripts/generate_m6_bank.py` | `path-events-m4` |

Release recipe 應明確指定 backend。

## Output level

`HybridRIRConfig.output_mode` 支援：

- `calibrated`：保留 source level、distance、receiver calibration 與 cross-room
  gain，peak 可能大於 1；
- `peak_normalized`：把整個 item 正規化到指定 peak。

Mixture normalization 是另一個 downstream decision，必須另外記錄。

## 產生資料

```bash
python egs/rir_generation/generate_hybrid_rir.py --help
```

完整 recipe 請看 [RIR generation](../../egs/rir_generation/README.md)。產生的
experiment directories 不是 source files，不應加入版控。
