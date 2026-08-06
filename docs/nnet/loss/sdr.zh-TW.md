# puresound.nnet.loss.sdr

English version: [sdr.md](sdr.md)

時域的 SDR / SI-SNR 系列 loss。直接用 L1/MSE 對 waveform 做 regression,模型
可以靠「對齊 reference 的整體音量」取巧過關,而不是真的對齊「波形的形狀」——
這不是語音增強/分離真正在乎的事。`SDRLoss` 把這件事正規化掉:`zero_mean`
去掉 DC offset,`scaled`（見下）在量測殘差之前,先把 reference 投影到跟
estimate 最匹配的音量尺度上,所以 loss 對 `s1`、`s2` 之間固定的音量差是
invariant 的。下面的 `scaled` / `scale_dependent` / `source_aggregated` /
`sdr_max` 這幾個 flag,就是用同一份實作切出 SI-SNR / SD-SDR / SA-SDR / t-SDR
這一整個系列的變體。

## Class: `SDRLoss`

### Constructor

```python
SDRLoss(
    scaled: bool = True,
    scale_dependent: bool = False,
    zero_mean: bool = True,
    source_aggregated: bool = False,
    sdr_max: int = None,
    eps: float = 1e-8,
    reduction: bool = True,
    threshold: Optional[float] = None,
)
```

**Parameters:**
- `scaled` – 若為 `True`,在計算殘差之前,先把 reference 投影到 estimate 的
  最佳擬合尺度上（見下方 `forward`）。這就是讓這個 loss 具備 scale-*invariant*
  性質的地方（SI-SNR 裡的「SI」）。
- `scale_dependent` – 若為 `True`,殘差改成對著原始（未縮放）的 reference
  量測,而不是縮放過的版本。只有在 `scaled=True` 時才有作用——見 `forward`
  下方的說明。
- `zero_mean` – 在做任何事之前,先把兩個訊號各自的 per-utterance 平均值
  （DC offset）去掉。這是投影能代表「形狀」而不是「形狀加一個常數」的必要
  前提。
- `source_aggregated` – 在取比值之前,先跨 source 軸把能量加總起來
  （SA-SDR）。輸入需要是 3-D 的 `[batch, num_sources, length]`,而不是平常
  的 2-D `[batch, length]`——這一點由 `check_input_shape` 強制檢查。
- `sdr_max` – 若不是 `None`,套用一個 soft ceiling（"t-SDR"）:在取比值之前,
  把 `tau * target_norm`（`tau = 10 ** (-sdr_max / 10)`）加進 noise energy
  裡,讓算出來的 SDR 在殘差趨近於零時,漸近收斂在 `sdr_max` dB 附近,而不是
  發散。
- `eps` – 避免除以零的小常數。
- `reduction` – **是一個 bool,不是字串。** `True` 會回傳 `torch.mean(...)`
  （scalar）;`False` 回傳未經 reduce 的逐 row loss tensor。
- `threshold` – hard-example 篩選機制;見下方說明。

`SDRLoss.__init__` 同時會設定 `self.uses_inactive_labels = True`。這個 flag
是 `EncDecMaskBase.compute_loss`（`puresound/system/siso.py`）在找的
dispatch hook:任何設了 `uses_inactive_labels = True` 的 loss,呼叫時都會
多拿到一個 `inactive_labels` tensor,算法是
`target.abs().amax(dim=-1) == 0`——也就是 reference 整段都靜音的 row
（target-absent 的訓練樣本）。`SDRLoss` 怎麼處理它,見下方「Inactive rows」。

### `forward(s1, s2, inactive_labels=None) -> Tensor`

- `s1` – estimate/enhanced 訊號,形狀 `[batch, length]`,若
  `source_aggregated=True` 則是 `[batch, num_sources, length]`。
- `s2` – reference（clean target）訊號,形狀與 `s1` 相同。
- `inactive_labels` – 選填的 bool tensor `[batch]`;見「Inactive rows」。

對每個（active）row,在 `s1`/`s2` 做完選用的 zero-mean 之後:

```
s_target = <s1, s2> / (<s2, s2> + eps) * s2   if scaled else s2
e_noise  = s1 - s2                             if scale_dependent else s1 - s_target
target_norm = <s_target, s_target>
noise_norm  = <e_noise, e_noise> [+ tau * target_norm, if sdr_max is set]
snr = -10 * log10(target_norm / (noise_norm + eps) + eps)     # 取負號:這是 loss
```

（上面的 `<a, b>` 就是這個 module 的 `l2_norm(a, b)` 這個 helper——雖然叫
這個名字,它做的其實就是 `sum(a * b, dim=-1, keepdim=True)`,是內積;只有在
以 `l2_norm(x, x)` 呼叫時,才等於平方後的 L2 norm。）當
`source_aggregated=True` 時,同一套公式會先對 source 軸把能量加總
（`target_norm.sum(dim=-1)` / `noise_norm.sum(dim=-1)`）,而不是逐 source
分開算。

**`scaled` 與 `scale_dependent` 的關係:** `scale_dependent` 只有在
`scaled=True` 時才會改變任何結果。當 `scaled=False` 時,`s_target` 本來就
等於 `s2`,所以不管 `scale_dependent` 是什麼值,`e_noise` 都是
`s1 - s2`——這個 flag 是無作用的。讀 config 的時候這點很重要:
`egs/voice_isolate/config/train_dpcrn.yaml` 設的是 `scaled: False,
scale_dependent: True`,依這條規則,這裡的 `scale_dependent: True` 不起
作用;實際行為完全由 `scaled: False` 決定（見「Config usage」）。

**Returns:** 負的 SDR/SI-SNR 指標值（把 loss 最小化,就是把這個指標最大化),
若 `reduction=True` 會 reduce 成一個 scalar mean,否則回傳逐 row 的 tensor。

### Inactive rows:`inactive_sdr_loss`

普通的 SDR 在 reference 整段靜音的 row 上是無定義的——`target_norm`（數值上）
是零,所以 `10 * log10(target_norm / noise_norm)` 會發散到 `-inf`。`forward`
沒有把這個特例塞進主公式裡處理,而是在 `inactive_labels` 標出至少一個 row
時把 batch 拆開:active 的 row 走上面那套流程;inactive 的 row 改用
module-level 的 `inactive_sdr_loss(s1, s2, reduction=False)` 函式計分,然後
兩邊逐 row 的結果會被串接起來（`torch.cat([snr, inactive_loss], dim=0)`),
才套用最外層的 `reduction`。

**這不是排除。** 這份文件先前的版本曾把 inactive rows 描述成會被排除在
loss 之外,那是錯的。它們不會被排除,只是用不同公式計分,再併回同一個
mean 裡。

```python
def inactive_sdr_loss(s1, s2, reduction=True):
    # zero-mean, then:
    return 10 * log10(<s1, s1> + 0.01 * <s2, s2> + eps)
```

因為這些 row 上的 `s2` 依照定義就是靜音的,這個式子等於直接對 enhanced
訊號自己的能量 `<s1, s1>` 做懲罰——把它最小化,就是把模型推向「本來就該
安靜的地方保持安靜」,而不是去算一個無定義的比值。`0.01 * <s2, s2>` 是一個
下限,確保就算兩個訊號都恰好是零,log 也不會發散。注意這個值**沒有**像
active row 的 `snr` 那樣取負號——它本來的意義就是「數值愈小愈好」（洩漏的
能量愈少愈好),所以直接跟已經取過負號的 active row 項放進同一個 mean 裡,
數值意義是一致的。

### Hard-example 篩選機制（`threshold`）

設了 `threshold` 之後,逐 row loss 已經**低於** `threshold` 的 row,會在
reduce 之前被丟掉:

```python
snr_to_keep = snr[snr > self.threshold]
if snr_to_keep.nelement() > 0:
    snr = snr_to_keep.view(-1, 1)
```

也就是說,梯度只花在還「值得推」的 row 上。但如果整個 batch 裡**每一個**
row 都已經達標,`snr_to_keep` 會是空的,這時篩選機制會直接跳過——`forward`
會退回使用完整、未篩選的 batch,而不是去對一個空 tensor 做 reduce。

### 常見變體與 `init_mode`

```python
@classmethod
def init_mode(cls, loss_func: str = "sisnr", reduction: bool = True, threshold: Optional[float] = None) -> "SDRLoss"
```

一個依名稱選預設組合的另一種 constructor。它**沒辦法**透過 recipe 的
`loss_func[].type` 取用（package 裡只 export 了 `SDRLoss` 這個 class 本身;
`init_mode` 是掛在它上面的一個 method,不是一個 top-level 名稱),而且這個
repo 裡目前沒有任何 recipe 呼叫它——用 grep 找,只有 `sdr.py` 自己裡面
才有引用。Recipe 實際上都是直接呼叫 `SDRLoss(...)`,自己明確給 keyword
arguments（見「Config usage」)。以下是從目前原始碼追出來的預設組合表:

| `loss_func` 名稱 | `scaled` | `scale_dependent` | `source_aggregated` | `sdr_max` |
|---|---|---|---|---|
| `sisnr` | `True` | `False` | `False` | `None` |
| `sdsdr` | `True` | `True` | `False` | `None` |
| `sdr` | `True` ¹ | `False` | `False` | `None` |
| `tsdr` | `False` | `False` | `False` | `30` |
| `tsdr50` | `False` | `False` | `False` | `50` |
| `sasdr` | `False` | `False` | `True` | `None` |
| `sasisnr` | `False` ¹ | `False` | `True` | `None` |
| `satsdr` | `False` | `False` | `True` | `30` |

¹ **已知的小瑕疵,另案追蹤中**——不在這次文件整理裡修。`init_mode` 是這樣
決定 `scaled` 的:

```python
if loss_func == "sisnr" or loss_func in "sdsdr" or loss_func == "sasisdr":
    scaled = True
```

這一行裡有兩件事跟名字暗示的不一樣,而且兩者都出在同一行上:

- `loss_func in "sdsdr"` 是一個**寫反的 containment 檢查**:Python 會把它
  讀成「字串 `loss_func` 是不是字面字串 `'sdsdr'` 的子字串」,而不是
  「`loss_func` 是否等於 `'sdsdr'`」。因為 `"sdr"` 剛好是 `"sdsdr"` 的子字串
  （`"sdsdr"[2:5] == "sdr"`),所以 `sdr` 這個 preset 也會被悄悄設成
  `scaled=True`——結果就是現在 `SDRLoss.init_mode("sdr")` 跟
  `SDRLoss.init_mode("sisnr")` 建出來的是一模一樣的物件,而不是讓 `"sdr"`
  成為一個單純、非 scale-invariant 的 SDR。
- 第三個判斷式比對的字面字串是 `"sasisdr"`,而不是真正的 mode 名稱
  `"sasisnr"`（`d`/`n` 兩個字母被寫反了)。`"sasisdr"` 從來就不是一個合法的
  `loss_func` 值——它不在 `init_mode` 一開始檢查的允許清單裡——所以這個
  判斷式永遠不會成立,對它原本大概想接住的那個 mode 來說,是一段死代碼。
  實際效果:`SDRLoss.init_mode("sasisnr")` 拿到的是 `scaled=False`,而不是
  它名字裡「SI」暗示的 `True`,導致它現在跟 `SDRLoss.init_mode("sasdr")`
  一模一樣。

這個問題只影響本節說明的 `init_mode(...)` 這個 convenience constructor;
這裡記錄它,是為了讓上面那張預設組合表不會看起來莫名其妙,而不是把它當成
刻意設計的行為來介紹。

### Config usage

```yaml
# egs/voice_isolate/config/train_dpcrn.yaml（現役 recipe）
loss_func:
  - type: SDRLoss
    weighted: 1.
    args:
      scaled: False
      scale_dependent: True   # 這裡不起作用 -- scaled 是 False,見上面的說明
      zero_mean: True
      source_aggregated: False
      sdr_max: 50              # soft ceiling,約 50 dB
      threshold: -50           # 已經衝過那個 ceiling 的 row 就不再推了
```

這裡用的是原始 constructor 加上明確的 kwargs,`(scaled=False,
scale_dependent=True)` 這組組合,上面任何一個 `init_mode` preset 都做不出
來——這個 recipe 的行為並不依賴 `init_mode` 的自動預設值,也就不受它那個
瑕疵影響。

## SI-SNR 公式（`scaled=True, scale_dependent=False` 這個情況)

$$\text{loss} = -10 \log_{10} \frac{\|\text{proj}(\hat{s})\|^2}{\|\hat{s} - \text{proj}(\hat{s})\|^2}, \qquad \text{proj}(\hat{s}) = \frac{\langle \hat{s}, s \rangle}{\langle s, s \rangle}\, s$$

其中 $\hat s$ = `s1`（enhanced),$s$ = `s2`（reference),兩者都是做完選用的
zero-mean 之後的版本。

## Module-level helpers

以下都沒辦法透過 `loss_func[].type` 取用——`puresound/nnet/loss/__init__.py`
只 import 了 `SDRLoss`。

- `si_snr(s1, s2, eps=1e-8, reduction=True)` – 跟
  `SDRLoss(scaled=True, scale_dependent=False, source_aggregated=False)`
  算的是同一套東西,只是寫成一個獨立函式。這是真的有在用的程式碼:
  `puresound/metrics.py` 直接 import 它,拿來算 `sisnr` /
  `sisnr_imp`(相對 noisy mixture 的 SI-SNR improvement)這兩個**評估用**
  指標。
- `inactive_sdr_loss(s1, s2, reduction=True)` – 見上文。
- `l2_norm(s1, s2)` – 整個 module 都在用的內積 helper。
- `attenuation_ratio(s1, s2, mask, reduction=True)` – 只針對訊號裡非
  target 的部分（`mask == 0`)量測 per-utterance 的 dB 衰減量,也就是系統
  在該安靜的地方壓得多乾淨。目前 repo 裡沒有其他地方呼叫它。

## Example

```python
from puresound.nnet.loss.sdr import SDRLoss

sisnr_loss = SDRLoss(scaled=True, scale_dependent=False, zero_mean=True)
loss = sisnr_loss(enhanced_wav, clean_wav)

# target-absent 的 row 會交給 inactive_sdr_loss 計分,不會被丟掉:
inactive_labels = clean_wav.abs().amax(dim=-1) == 0
loss = sisnr_loss(enhanced_wav, clean_wav, inactive_labels=inactive_labels)
```
