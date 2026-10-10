# puresound.audio.io

English version: [io.md](io.md)

`AudioIO` 透過 `soundfile`（libsndfile）讀寫音訊檔，載入時可選擇重新取樣與
RMS 位準調整。所有方法都是 `@staticmethod`：請呼叫 `AudioIO.open(...)`，不要
`AudioIO().open(...)`。建構子存在，但只存一個沒有任何方法會讀的 `verbose`。

## `audio_info(f_path) -> (sample_rate, total_samples, total_seconds, num_channels)`

用 `soundfile.info` 只讀檔頭，不讀波形。
`total_seconds = round(frames / sample_rate, 2)`。回傳值是依此順序的普通 tuple。

## `open(f_path, resample_to=None, normalized=False, target_lvl=None, verbose=False) -> (wav, sr)`

1. 以 float32、`always_2d=True` 讀檔，回傳 `wav: [C, L]`（聲道在前）。PCM 檔
   的值落在 [-1, 1]。
2. 若設定了 `resample_to` 且與檔案取樣率不同，用
   [`dsp.wav_resampling(..., backend="sox")`](dsp.zh-TW.md) 這個確定性的取樣率
   轉換器轉換，回傳的 `sr` 就是 `resample_to`。
3. 若設定了 `target_lvl`（dBFS）且 `normalized` 為 false，用
   [`volume.rescale_waveform`](volume.zh-TW.md)（`amp_type="rms"`、
   `scale="dB"`）把 RMS 調到該位準：

   ```
   wav ← wav / (rms(wav) + 1e-14) · 10^(target_lvl / 20)
   ```

   位準調整在重新取樣之後，所以位準是在輸出取樣率下量的。

`normalized=True` 會關掉 `target_lvl` 路徑，並把每個聲道正規化到平均振幅 1
（`volume.normalize_waveform(..., amp_type="avg")`）。沒有任何呼叫端使用它；調位準請用
`target_lvl`。

**使用者：** dataset 層以 `resample_to=dataset.target_sample_rate`、
`target_lvl=dataset.gain_normalized_to`（YAML 留空 = 不調位準）載入每段語音；
`AudioEffectAugmentor` 用它載入噪音與資料夾 RIR；評估系統與 inference processor
用 `resample_to` 載入檔案。

## `save(wav, f_path, sr, subtype="PCM_16", **kwargs)`

以 `soundfile.write` 寫檔。注意參數順序：波形在前、路徑在後，和 `open` 相反。
1-D 的 `wav` 視為 `[1, L]`；tensor 會 detach、移到 CPU、轉置成 libsndfile 的
`[L, C]`。`subtype` 預設 16-bit PCM（超出 [-1, 1] 的樣本會被截斷）；要保留
32-bit float 請傳 `"FLOAT"`。其餘關鍵字參數轉給 `soundfile.write`。

## `cut_audio(wav, sr, length_s, padding=False) -> (wav, offset, end_offset)`

把 `[C, L]` tensor 隨機裁成 `target_len = sr · length_s` 個樣本：

| 輸入長度 | 結果 |
| --- | --- |
| `L > target_len` | 在 `[0, L - target_len]` 取隨機起點，切成剛好 `target_len` |
| `L <= target_len`，`padding=True` | 在尾端補零到 `target_len` |
| `L <= target_len`，`padding=False` | 原樣回傳，比 `target_len` 短 |

`sr · length_s` 會被截成整數樣本數。起點由 Python 的 `random` 抽。

## `audio_cut(wav, sr, length_s) -> (wav, (offset, end_offset))`

把 1-D 輸入升成 `[1, L]` 後呼叫 `padding=True` 的 `cut_audio`。

訓練 dataset 不使用這兩個裁切函式，而是用自己的 `align_audio_list` 對齊長度
（見 [dynamic_base](../../architecture/dataset/dynamic_base.zh-TW.md)）。

## 範例

```python
from puresound.audio.io import AudioIO

wav, sr = AudioIO.open("speech.wav", resample_to=16000, target_lvl=-28.0)
sample_rate, total_samples, duration_s, num_channels = AudioIO.audio_info("speech.wav")
AudioIO.save(wav, "output.wav", sr)                   # 16-bit PCM
AudioIO.save(wav, "output_f32.wav", sr, subtype="FLOAT")
```
