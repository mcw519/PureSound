# Multiband FDN — `puresound.audio.rir.render.multiband_fdn`

English version: `multiband_fdn.md`

決定性（deterministic）、被動（passive）的 feedback delay network，可對每個
octave 分別控制 RT60。這是 M4 backend 背後的晚場（late-field）引擎；設計對照
詳見 [`rir_realism_algorithm.zh-TW.md`](rir_realism_algorithm.zh-TW.md) §4.3。

## Design contract（設計契約）

- `design_multiband_fdn(...)` → `MultibandFDNDesign`；
  `render_multiband_fdn(...)` / `render_multiband_fdn_impulse(...)` →
  `MultibandFDNRender`。Policy 字串：`MULTIBAND_FDN_POLICY`。
- Feedback matrix：`randomized_hadamard_matrix`（正交矩陣 → 被動迴路）。
- Delay：`select_prime_delay_lengths`（互質、有密度但不會有週期性）。
- 每個頻帶的衰減：`delay_proportional_loop_gains` 會針對每條迴路的長度，
  精確實現各 octave 的目標 RT60。
- Filterbank：串接式二元分割（cascaded binary split），端點完整
  （endpoint-complete）——最高頻帶是一路 highpass 到 Nyquist，所以晚場能量
  在 fs/2 以下不會有頻譜空洞（`fdn_filterbank_power_response` 會驗證其平坦度）。
- 一切都有 seed；相同輸入會 render 出逐位元組相同（byte-identical）的尾段。

## Diagnostics（診斷工具）

`analyze_fdn_coloration` 回報 render 出來的尾段有多少 modal coloration；
`puresound.audio.rir.metrics.analyze_multiband_late_field` 則檢查一段尾段
是否符合它自己每個 octave 的目標值。

## Caveat（但書）

FDN 只會忠實實現使用者給它的目標；目標值的品質是呼叫端自己的責任——
Sabine 推導出的目標值有哪些已知的量測極限，見 。
