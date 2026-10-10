# puresound.metrics

English version: [metrics.md](metrics.md)

語音增強與分離用的客觀 metric：

- `puresound.metrics.Metrics` —— 以參考訊號為基準的品質與可懂度分數、免參考的
  DNSMOS，以及幀標籤 F1。`--scoring` 執行時透過 `puresound/system/runner.py` 的
  `WAVEFORM_METRICS` 讀取它們；閘門的 `reference` 與 `noreference` 階段則直接呼叫。
- `puresound.evaluation.spectral` —— 兩個衡量模型 bottleneck 丟掉了什麼的量（諧波
  對比、瞬態相關），由 `reference` 階段回報。
- `puresound.evaluation.tools.wer` —— 字錯誤率及其刪除成分，由 `wer` 階段使用。

評測協定、配對統計與判定見 [evaluation](../usage/evaluation.zh-TW.md)。

## Class: `Metrics`

Static methods。不論 `np.array` 型別標注怎麼寫，輸入都是 shape 為 `[1, T]`（或
`[C, T]`，只評 channel 0）的 `torch.Tensor`。

### 輸入處理：`check_shape`

```python
Metrics.check_shape(
    clean: torch.Tensor,
    enhanced: torch.Tensor,
    retun_as_tensor: bool = False,   # 參數的實際拼法
) -> Tuple[np.ndarray, np.ndarray]   # retun_as_tensor=True 時為 (Tensor, Tensor)
```

除了 `dnsmos_p835` 之外，每個 method 都會先把輸入交給它：

1. 第一軸大小不是 1 時，只保留該軸的 index 0（`x[0, ...]`）。因此 1-D 的 `[T]`
   tensor 會塌成它的第一個 sample——請傳 `[1, T]`。
2. Squeeze 成 1-D。
3. 從開頭把較長的訊號裁到與較短者等長（不做 cross-correlation 對齊）。
4. 轉成 NumPy，並**各自以自己的** `abs().max()` **做 peak-normalize**。

所以分數與輸出音量無關。全零輸入會在第 4 步除以零，在下游變成 NaN。

### 以參考訊號為基準的分數

| method | 計算內容 | 回傳 |
| --- | --- | --- |
| `pesq_wb(clean, enhanced)` | wide-band PESQ（ITU-T P.862.2），經 `pesq` 套件；取樣率固定為 16000 | MOS-LQO，越高越好 |
| `pesq_nb(clean, enhanced)` | narrow-band PESQ（ITU-T P.862）；取樣率固定為 8000，輸入必須已是 8 kHz | MOS-LQO，越高越好 |
| `stoi(clean, enhanced, sr=16000)` | STOI（Taal et al., 2011），經 `pystoi` | `[0, 1]`，越高越好 |
| `estoi(clean, enhanced, sr=16000)` | extended STOI（Jensen & Taal, 2016），`stoi(..., extended=True)` | `[0, 1]`，越高越好 |
| `bss_sdr(clean, enhanced)` | BSS Eval SDR（Vincent et al., 2006），`mir_eval.separation.bss_eval_sources(clean, enhanced, False)[0][0]`，單一來源、不搜尋排列 | dB |
| `sisnr(clean, enhanced)` | SI-SNR（見下） | dB，float |
| `sisnr_imp(clean, enhanced, noisy)` | `SI-SNR(enhanced, clean) − SI-SNR(noisy, clean)`；`clean` 分別與兩者各自對齊 | dB，float |
| `noise_reduction(noisy, enhanced)` | 在 peak-normalize 後的一對訊號上算 `10 log10(Σ enhanced² / Σ noisy²)`——是正規化後功率的比值，不是絕對能量的比值 | `Tensor [1]` |

SI-SNR（Le Roux et al., 2019），取自 `puresound.nnet.loss.sdr.si_snr`，兩個訊號都先
去除平均：

```
s_target = <ŝ, s> / <s, s> · s
e        = ŝ − s_target
SI-SNR   = 10 log10(‖s_target‖² / ‖e‖²)
```

### `dnsmos_p835`

```python
Metrics.dnsmos_p835(
    clean: torch.Tensor,           # 不使用；沒有參考訊號的呼叫端傳 None
    enhanced: torch.Tensor,
    sr: int = 16000,
    personalized: bool = False,
    num_threads: int | None = None,  # onnxruntime intra/inter-op 執行緒數
) -> Dict[str, float]
# {"dnsmos_p808": ..., "dnsmos_sig": ..., "dnsmos_bak": ..., "dnsmos_ovr": ...}
```

免參考的 DNSMOS P.835（Reddy et al., 2022）加上 P.808 整體分數，在 CPU 上經
`torchmetrics.audio.dnsmos.DeepNoiseSuppressionMeanOpinionScore` 計算。`enhanced`
會被 detach、轉成 float、化為單一 channel（先 squeeze 掉前導的大小 1 軸，再取
channel 0），並 clamp 到 `[-1, 1]`；它**不會**做 peak-normalize。每個
`(sr, personalized, num_threads)` 組合快取一個 metric 實例。

`num_threads=None` 使用所有核心，適合單一行程。worker pool 必須傳 `1`：在 worker 裡
`torch.set_num_threads(1)` 管不到 onnxruntime，否則每個 worker 都會開一整組執行緒，
把機器超額訂閱。`puresound.evaluation.tools.noreference` 傳的就是 1。沒有安裝
torchmetrics 的 audio 額外相依時，呼叫會丟出帶安裝提示（`uv sync`）的
`ModuleNotFoundError`。

### `f1_score`

```python
Metrics.f1_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> Dict[str, float]
# {"accuracy": ..., "precision": ..., "recall": ..., "f1_score": ...}
```

二元幀標籤（非零 = 正例），計數為 TP / TN / FP / FN：
`precision = TP / (TP + FP + 1e-7)`、`recall = TP / (TP + FN + 1e-7)`、
`F1 = 2PR / (P + R + 1e-7)` 並 clamp 到 `[1e-7, 1 − 1e-7]`。標籤會經過
`check_shape`，所以沒有任何正例幀的 tensor 會在那裡除以零。

### 範例

```python
from puresound.metrics import Metrics

clean, enhanced = clean_wav.view(1, -1), enhanced_wav.view(1, -1)
sisnr_val = Metrics.sisnr(clean, enhanced)
pesq_val  = Metrics.pesq_wb(clean, enhanced)
stoi_val  = Metrics.stoi(clean, enhanced, sr=16000)
dnsmos    = Metrics.dnsmos_p835(None, enhanced)
```

## Bottleneck 量測：`puresound.evaluation.spectral`

兩者都在 Hann 窗、置中的 dB STFT（`20 log10 |X|`）上計算，並把增強後訊號與同一個
item 的 target 比較。

```python
harmonic_contrast_db(wav, sample_rate=16000, *, n_fft=512, hop=160,
                     band_hz=(100.0, 2000.0), active_percentile=0.6) -> float
harmonic_contrast_gap_db(enhanced, target, sample_rate=16000, **kwargs) -> float
transient_correlation(enhanced, target, sample_rate=16000, *, n_fft=128, hop=32) -> float
```

- **`harmonic_contrast_db`** 保留平均 log-magnitude 不低於 `active_percentile`
  分位數的幀，在 `band_hz` 內找出頻譜的局部峰與谷（比左右兩鄰都大或都小），再把
  `mean(peaks) − mean(valleys)` 在這些幀上取平均。
- **`harmonic_contrast_gap_db`** = `contrast(enhanced) − contrast(target)`，
  `reference` 階段以 `harmonic_gap_db` 回報。負值代表諧波梳狀結構被抹平。
- **`transient_correlation`** 取每個訊號在 8 ms / 2 ms（16 kHz 下）的逐幀 log 能量
  （各 bin 的 dB 平均），做一階差分，回傳兩條斜率的 Pearson 相關，範圍 `[-1, 1]`。
  以 `transient_corr` 回報。

沒有東西可量時回傳 NaN：幀數少於 3（諧波）或 4（瞬態）、頻帶窄於 5 個 bin、找不到
峰谷配對，或斜率變異為零。

**為什麼。** 有聲語音是一排諧波梳。頻率解析度太粗的 bottleneck 產生的 mask 跟不上這排
梳子，峰與谷就會互相靠攏；能量比會把「去掉的噪音」與「被抹平的語音」加在一起，這個量把
兩者分開。回報差值而非絕對對比，是因為絕對值取決於講者與語句。瞬態（鍵盤、關門）只有
幾毫秒；時間解析度太粗的 bottleneck 會抹掉它們的邊緣。相關的對象是斜率而不是包絡，因為
響度相同、起音銳利度不同的兩個訊號在包絡上仍然一致；窗也比模型計算的任何幀都細，所以
能看到模型自身幀率看不到的抹平。

## 字錯誤率：`puresound.evaluation.tools.wer`

```python
normalise(text: str) -> str                  # 轉小寫、"-" -> " "、去掉 [a-z0-9' ] 以外的字元
edit_counts(reference: str, hypothesis: str) -> dict   # {"sub", "del", "ins", "hit", "ref_words"}
rates(counts: dict) -> dict                  # {"wer", "del", "ins", "sub"}
loop_count(rows, ratio: float = LOOP_RATIO) -> int     # LOOP_RATIO = 1.5
```

`edit_counts` 是以字為單位、帶回溯的 Levenshtein 對齊（平手時依序偏好 match/替換、
刪除、插入）。令 `N = max(ref_words, 1)`：

```
WER = (S + D + I) / N      del = D / N      ins = I / N      sub = S / N
```

`loop_count` 計算字數超過 `ratio × ref_words` 的 hypothesis：辨識器解析不了某段時會
重複同一句，而單一個這種 item 的插入率就可能主導整個語料的平均。

**為什麼。** 刪除單獨回報，因為它是有方向性的失敗：替換與插入也可能來自辨識器本身，但
對同一份參考文字刪除率上升，就是模型把語音拿掉了。參考文字是語料的原始轉寫，絕不用
辨識器在乾淨音訊上的輸出——那只是在量它和自己的一致程度。
