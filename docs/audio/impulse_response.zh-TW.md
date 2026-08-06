# puresound.audio.impulse_response

English version: `impulse_response.md`

Room Impulse Response（RIR）卷積與隨機 IIR 濾波工具。

> **注意：** 檔案本身 `impulse_response.py` 拼寫正確；typo 藏在下一層——
> `wav_apply_rir` 的 RIR 參數命名為 `impaulse`，這個拼字錯誤是為了向後相容
> 才保留下來的。

## Functions

### `compute_drr_db(rir: Tensor, sample_rate: int, direct_window_ms: float = 2.5) -> float`

`puresound.audio.rir.metrics.compute_drr_db` 的簡單 re-export。Direct path
＝落在 `[peak, peak + direct_window_ms]` 區間內的能量；這個窗之後的部分都當作
殘響尾段（reverberant tail）。當尾段沒有能量時（例如 anechoic 或已經被截斷的
RIR）會回傳 `+inf`。

---

### `wav_apply_rir(wav: Tensor, impaulse: Tensor, sample_rate: int, rir_mode: str = "full") -> Tensor`

將波形與 Room Impulse Response 卷積，藉此模擬殘響。

**參數：**
- `wav` – 乾淨語音波形 tensor，形狀 `[channels, T]`（必須是 2 維；若傳入
  單一維度 `[T]` 的 tensor，內部 `wav_ch, _ = wav.shape` 這行 unpack 會
  直接丟出例外）
- `impaulse` – Room impulse response tensor，形狀 `[channels, T_rir]`
  （同樣要求是 2 維）
- `sample_rate` – 取樣率（Hz），用來決定 `"direct"`／`"early"` 窗的長度
- `rir_mode` – 殘響模式：
  - `"full"` – 完整卷積；輸出保留所有晚期反射
  - `"direct"` – 把 RIR 截到 `[peak, peak + 6 ms]`
  - `"early"` – 把 RIR 截到 `[peak, peak + 50 ms]`

**回傳：** 殘響波形，修剪回 `wav` 原本的長度，並且時間對齊到 RIR 的 peak
sample（也就是 direct arrival），而不是從原始卷積結果的起點算起。

若 `impaulse` 只有一個 channel，每個 `wav` channel 會各自獨立與它卷積
（[N, T] 進、[N, T] 出）。若 `impaulse` 有多個 channel，`wav` 就必須是單一
channel，每個 RIR channel 各自產生一個輸出 channel（[1, T] 進、
[C_rir, T] 出）——這是 mic-array 的情境。

**注意事項：**
- 使用 FFT-based 卷積（`fftconvolve`）。
- 在 `"direct"` 與 `"early"` 模式下，RIR 會先被窗化*才*進行卷積，接著才做
  peak 正規化（peak-normalize）。因為窗一律從 peak 開始算，同一支 RIR 在
  三種 `rir_mode` 下套用的縮放係數都相同——所以用同一支 RIR 做出的
  `"early"` 模式乾淨目標，與 `"full"` 模式的帶噪混音，會共享相同的
  direct-path 音量；差別只在晚期殘響的能量。
- **副作用：** bank 裡的 RIR 在磁碟上帶有實際的 `1/r` 距離增益，channel
  間的相對音量關係本來完整保留，但這個 per-channel peak 正規化會把它抹掉
  ——混音因此不會繼承距離相關的音量；能區分近／遠場的只剩下 DRR、衰減
  形狀與頻譜傾斜（spectral tilt）。Recipe 對 SIR 的處理（`mix_mode:
  physical`、hard-SIR 範圍）都是建立在這個假設之上；改動這個正規化方式，
  這些假設也會一併改變。

---

### `rand_add_2nd_filter_response(wav: Tensor, a: Optional[Tensor] = None, b: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]`

套用一個隨機二階 IIR（biquad）濾波器，模擬麥克風或訊號鏈路造成的頻率染色
[1]。

**參數：**
- `wav` – 輸入波形 tensor
- `a`, `b` – 選擇性、預先取樣好的分母／分子係數，各自形如 `[1, x1, x2]`。
  省略時，兩組 `x1, x2` 會各自獨立均勻取樣自 `[-3/8, 3/8]`。

**回傳：** `(filtered_wav, a, b)`——取樣到的係數會一併回傳，讓呼叫者可以把
「同一個」隨機濾波器套用到第二個相關訊號上（例如讓混音與其乾淨目標經過
一模一樣的染色）。

**注意事項：**
- 沒有 `sample_rate` 參數，也沒有 frequency／Q／gain 的設計步驟：係數是
  直接在 biquad 的 `a`／`b` 係數空間中取樣，而不是從
  highpass／lowpass／peaking／shelf 這類參數化推導出來的。

參考文獻：[1] *A Hybrid DSP/Deep Learning Approach to Real-Time Full-Band
Speech Enhancement*.

## Example

```python
from puresound.audio.impulse_response import wav_apply_rir, rand_add_2nd_filter_response
from puresound.audio.io import AudioIO

rir, _ = AudioIO.open("rir.wav")
clean, sr = AudioIO.open("clean.wav", target_sr=16000)

reverberant = wav_apply_rir(clean, rir, sr, rir_mode="early")
colored, a, b = rand_add_2nd_filter_response(clean)
```
