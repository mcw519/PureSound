# puresound.nnet.lobe.dsp

English version: [dsp.md](dsp.md)

以 STFT 上逐 bin 增益表示的 parametric EQ：一串 biquad filter 在建構時算一次，
其幅度響應成為一條（可選擇可訓練的）增益曲線。

## Class: `FrequencyEQLayer`

```python
FrequencyEQLayer(
    n_fft: int = 512,                 # 增益曲線有 n_fft // 2 + 1 個 bin
    sample_rate: int = 16000,
    eq_band_gain: Tuple[float] = (0.5, 5.5, -3.25, -2.5, -4, -4, -4.5),        # dB，每項一個 peaking band
    eq_band_cutoff: Tuple[float] = (500, 1000, 1500, 2500, 3500, 5500, 6000),  # Hz，中心頻率
    eq_band_q_factor: Tuple[float] = (0.707, 0.707, 0.707, 0.707, 0.707, 0.707, 0.707),
    low_shelf_gain_dB: float = 0.0,
    low_shelf_cutoff_freq: float = 80,
    low_shelf_q_factor: float = 0.707,
    high_shelf_gain_dB: float = 0.0,
    high_shelf_cutoff_freq: float = 7800,
    high_shelf_q_factor: float = 0.707,
    trainable: bool = True,           # peq 為 nn.Parameter；否則為 buffer
)
```

peaking band 的數量是 `len(eq_band_gain)`；`eq_band_cutoff` 與 `eq_band_q_factor`
至少要一樣長。

### 計算內容

low shelf、每個 peaking band 與 high shelf 的 `(b_k, a_k)` 由
`puresound.audio.dsp.get_biquad_params` 產生。各級串接後只保留幅度：

```
H(f) = | prod_k  rfft(b_k, n_fft)(f) / rfft(a_k, n_fft)(f) |      # [n_fft // 2 + 1]
peq  = H.view(-1, 1)                                               # [n_fft // 2 + 1, 1]
```

`forward(x [..., n_fft // 2 + 1, T]) -> x * peq`，shape 不變。輸入是頻域 tensor，
不是波形。

`get_args`（property）以 dict 回傳 constructor 參數，讓這一層能以不同的
`trainable` 重建。

## 在 recipe 中的用法

與 `encoder` 並列的 `freq_eq` 區塊會建出這一層；`puresound.recipes` 把它當作
`peq_module` 傳給 `FeatureEncoder`。`FeatureEncoder` 以 `get_args` 重建它，並使用
feature 區塊自己的 `trainable`，套用在 encoder 輸出的複數 STFT 上（以
`[N, 2, F, T]` 形式、在丟掉 DC bin 之前）。見 [features](../features.zh-TW.md)。

```yaml
model:
  freq_eq:
    type: FrequencyEQLayer
    eq_args:
      n_fft: 512
      sample_rate: 16000
      eq_band_gain:   [2.5, 5.5, 3.25, 2.5, -2, -4, -8]
      eq_band_cutoff: [500, 1000, 1500, 2500, 3500, 5500, 7500]
```

## 設計說明

- 只有串接後的增益曲線 `peq` 會被訓練。band 的增益、截止頻率與 Q 值只決定初始值；
  之後任何曲線形狀都可達到。
- 響應只取幅度，所以每個 bin 的實部與虛部乘上同一個實數增益，這一層不改變相位。
- biquad 只算一次並存下曲線，forward 只是一次 broadcast 乘法。
