# noise_suppression —— 評測定義與紀錄

English: [`README.md`](README.md)

這個目錄進版控，而且**不放任何音訊**。它裝的是每一道評測關卡是什麼，以及每一個已發布的
checkpoint 與對照的外部系統拿到什麼數字。這些數字所依據的音訊放在不進版控的
`data_report/`；重建那些音訊的指令，是 [`stages.zh-TW.md`](stages.zh-TW.md) 裡每道關卡
定義的一部分。

把數字進版控就是重點。一個只存在於對話記錄或 run 目錄裡的版本比較，之後會從頭再吵一次，
而且通常換了一套協定。

## 佈局

```
benchmarks/
  README.md            這份檔案 —— 紀錄規格
  stages.md            每道閘門關卡是什麼、它能分辨什麼
  records/<tag>.json   一個評過分的 checkpoint（或外部系統）一個檔案
```

`run_full_gate.sh <tag> ...` 會寫出 `records/<tag>.json`；怎麼執行見
[`docs/usage/evaluation.zh-TW.md`](../../../docs/usage/evaluation.zh-TW.md)。

## 一筆紀錄必須帶的欄位

每筆紀錄是一個 JSON 物件，以下欄位缺一不可。缺任何一個，這筆紀錄就不是「不完整」，
而是無法用來比較。

| 欄位 | 為什麼是必填 |
| --- | --- |
| `tag` | 紀錄的名字，與檔名相同 |
| `checkpoint` | 被評分的是哪個檔案、第幾個 epoch（基線是 `unprocessed`，precomputed 音訊則是那個目錄） |
| `recipe` | 模型是用哪個 config **建**出來的 —— recipe 與 checkpoint 不合會靜默丟權重 |
| `chain_commit` | 評分時的 `git rev-parse --short HEAD`，dirty 要標 `+dirty`；凍結集的建集身分保存在該關卡的 `extra.set_provenance` |
| `inference` | 當時生效的旋鈕（`dry_blend`、各種 guard、`precomputed`）—— 沒有操作點的數字不可重現 |
| `stages` | 每道關卡一筆，各自帶角色、`n`、指標值、信賴區間與自己的判定 |
| `verdict` / `unresolved_gates` | 整體的 `pass` / `fail` / `no-resolution`，以及分辨不出的 gate 關卡 |
| `lineage` | 這個 run 在測什麼、上一階是誰、若被拔擢則寫 `released_as`——單一個 tag 撐不過一年 |

`evaluation.tools.collect` 會寫出 `lineage` 以外的所有欄位；`lineage` 在紀錄歸檔時手動
補上：`variable`（這個 run 改了什麼）、`parent`、`snr_range`，以及 checkpoint 被拔擢時的
`released_as` / `release_note`。

`no-resolution` 是一個真實的判定，而且一定要用。當一道關卡的信賴區間蓋過了你想宣稱
的差距，它就是什麼都沒量到；把它記成一場勝利，正是不可部署的版本被升上去的方式。

## 紀錄的命名

紀錄的檔名是 run 目錄名加上被評的 epoch，`<task>_<arch>_<variable>_<stage>_ep<N>`——例如
`ns_dpcrn-mamba_activebin_ft_ep1.json`。同一顆 checkpoint 在別的集上、或在別的操作點上另外
評的紀錄，加一個後綴（只在困難 WER 集上評的是 `_werhard`）。透過 `PRECOMPUTED_DIR` 評分的
公開系統命名為 `ext_<system>`。

版號只出現在一個地方：catalog id，在一顆 checkpoint 被拔擢時才指派。
`noise-suppression-dpcrn-mamba-v1` 就是紀錄 `ns_dpcrn-mamba_activebin_ft_ep1`；那個 run
從來沒有帶過 `v1`，而紀錄的 `lineage.released_as` 寫著 catalog id。改過名的紀錄會在
`lineage.previously_recorded_as` 保留舊名，它的 `checkpoint` / `recipe` 仍指向產物實際所在
的位置。

## 紀錄遵守的規則

1. **報絕對殘留，不要只報改善量。** 從一個很糟的起點拉出很大的 delta，不代表終點好。
2. **單一 checkpoint 不構成一次量測。** 同一個 run 相鄰 epoch 在某個指標上的擺幅，可能
   比要比較的版本差距還大。要評分整個 run 末尾的一組 checkpoint，報 block 統計量。
3. **每道關卡都要印 `n` 與區間。** 一個無法把候選版本和「不處理」分開的測試集是
   monitor，不是 gate；`stages.md` 要寫明它是哪一種。
4. **WER 讀的是對未處理混音的 delta**，用同一組切句，而且用強的辨識器。弱辨識器會
   把過度抑制藏起來。
5. **dB 差要在配對的 SNR 下比。** 跨測試集的絕對 dB 帶有測試集相依的 bias。
6. **綜合品質分數只是 monitor。** 報它是因為它對外可比，不是因為它能分辨我們的失敗
   模式 —— 一個什麼都沒做、也就沒有失真的 passthrough 可以拿到不錯的分數。
