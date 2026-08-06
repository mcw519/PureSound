# puresound.nnet.loss.vad

English version: [vad.md](vad.md)

Frame-activity 監督訊號:一個從 waveform 本身推導出 activity target
（`VADActivityLoss`),兩個監督一個明確的 backbone head（`VADHeadBCELoss`、
`BackgroundVADHeadBCELoss`),還有一個是通用的可微分 F1 指標（`F1_loss`)。

## Class: `VADActivityLoss`

對 frame-level 語音 activity 做的可微分 BCE。**純時域運算**:它用
`Tensor.unfold` 把 waveform 切成 frame,直接取每個 frame 的均方功率——這個
class 裡完全沒有 STFT/FFT（這份文件先前的版本曾描述成 STFT-based 的
framing,用了 `n_fft` / `win_length` 這些這個 class 根本沒有的參數;下面的
範例已經改用真正的 constructor)。

### Constructor

```python
VADActivityLoss(
    frame_length: int = 400,
    hop_length: int = 160,
    activity_threshold_db: float = -40.0,
    logit_scale: float = 0.25,
    false_positive_weight: float = 1.0,
    false_negative_weight: float = 1.0,
    require_vad_target: bool = False,
    eps: float = 1e-8,
)
```

**Parameters:**
- `frame_length` / `hop_length` – 對原始取樣點做 `unfold` framing 時用的
  window/hop（單位是取樣點,不是 FFT bin)。
- `activity_threshold_db` – 用來區分「active」跟「inactive」frame 的 dB
  門檻。這個門檻是相對什麼量測的,要看下面兩種模式的哪一種在運作。
- `logit_scale` – 把 `(enh_db - activity_threshold_db)` 縮放成一個 BCE
  logit;控制門檻附近的軟性判斷有多「陡」。
- `false_positive_weight` / `false_negative_weight` – BCE 的類別權重。
- `require_vad_target` – 若為 `True`,在沒有給 `vad_target` 時,`forward`
  會直接丟出 `ValueError`,而不是退回用 reference 推導 target。
- `eps` – 除法/log 的保護值。

### `forward(enh, ref, vad_target=None) -> Tensor`

兩個 waveform 都會先轉成 `[B, T]` 並對齊長度,再透過 `_frame_power` 做
framing（不足一個 frame 長度就先補零,`unfold(-1, frame_length,
hop_length)`,然後 `frames.square().mean(dim=-1)`——沒有 window function,
沒有 FFT)。依有沒有傳入 `vad_target`,分成兩種模式:

- **沒有外部 target**（預設情況;若 `require_vad_target=True` 則改為直接
  丟出例外):activity label 是從 `ref` *自己*的功率算出來的,用同樣的
  framing 方式,相對於這段 utterance 自己的 peak frame power 來判斷
  （`ref_db = 10*log10(ref_power / ref_power.amax())`)——這是一個會隨每段
  clip 音量自動調整的、utterance-相對的門檻。如果 reference 整段都是靜音,
  target 會被明確強制設成全部 inactive（而不是單靠有 eps 保護的除法自己
  算出剛好落在門檻以下)。
- **有外部 `vad_target`**（來自上游的 labeler,例如 recipe config 裡的
  `vad_label: {backend: silero}` 或 `{backend: energy}`):直接使用給定的
  label（補零/截斷到對齊 frame 數),而且——這是最容易被忽略的一點——
  enhanced 訊號的功率,量測基準改成一個**絕對的 0 dBFS 錨點**
  （`reference_power = ones`),而不是 reference 自己的 peak。原始碼裡的
  註解說明了原因:這樣可以讓 `activity_threshold_db` 變成一個絕對的 dBFS
  門檻,不受 reference 訊號本身音量/削波擾動影響,因為模型輸出被 clamp 在
  `[-1, 1]` 之間,所以 `enh_db` 一定落在 `(-inf, 0]` 之間。

不管是哪種模式,`enh` 的 frame 功率都會依該模式的 `reference_power` 轉成
dB,再轉成一個 BCE logit（`(enh_db - activity_threshold_db) * logit_scale`),
用 `false_positive_weight` / `false_negative_weight` 這兩個類別權重,透過
`binary_cross_entropy_with_logits` 對 `target_activity` 計分。

### Config usage

```yaml
# egs/noise_suppression/config/dpcrn.yaml（現役 recipe)
# VAD-driven objective: reduce background-speaker false activity while
# preserving clean foreground-active frames.
- type: VADActivityLoss
  weighted: 0.1
  args:
    frame_length: 800
    hop_length: 320
    activity_threshold_db: -40
    false_positive_weight: 2.0
    false_negative_weight: 1.0
```

### Example

```python
from puresound.nnet.loss.vad import VADActivityLoss

vad_loss = VADActivityLoss(frame_length=400, hop_length=160, activity_threshold_db=-40.0)
loss = vad_loss(enhanced_wav, clean_wav)
```

## Class: `VADHeadBCELoss`

對 backbone `VADHead`（見 [nnet.lobe.heads](../lobe/heads.md))算的 frame-level
BCE-with-logits,監督目標是資料集的 `vad_target`——跟上面的
`VADActivityLoss` 不同（那個是從 waveform 能量推導 activity),這個是監督
backbone 明確輸出的一個逐 frame logit head。會設定
`uses_vad_logits = True`,讓 `EncDecMaskBase.compute_loss`
（`puresound/system/siso.py`)把 `backbone.last_vad_logits` 路由進來。

```python
VADHeadBCELoss(false_positive_weight: float = 1.0, false_negative_weight: float = 1.0, balance_per_batch: bool = False)
```

`false_positive_weight` 會加重「靜音/遠場 frame 被誤判為 active」的權重。
`balance_per_batch` 會把兩個類別重新加權到 BCE 總質量相等,避免不平衡的
batch 靠著預測全 active 或全 silence 這種常數就能拿到獎勵（如果一個 batch
裡只出現一種類別,就沒有平衡可言,會退回原本的靜態權重)。`vad_logits`
跟 `vad_target` 之間的 frame 數,是用截斷成較短的一方來對齊的（原始碼裡的
註解提到,backbone 內部的 STFT framing 跟 label 的 framing 可能會差一個
frame)。目前隨附的 `train_dpcrn.yaml` recipe 沒有用到它;會用到的是
gated-bottleneck 實驗
`egs/voice_isolate/config/exp/train_dpcrn_gate.yaml`,搭配
`backbone_args.vad_head: {enabled: True, hidden: 128, kernel_t: 5}` 一起用。

## Class: `BackgroundVADHeadBCELoss`

```python
class BackgroundVADHeadBCELoss(VADHeadBCELoss):
    uses_vad_logits = False
    uses_background_vad_logits = True
```

它**沒有覆寫 `forward`**——跟 `VADHeadBCELoss` 是完全相同、會真的
backpropagate 的 BCE 運算,只是（透過上面這兩個 dispatch flag)改接到
`background_vad_target` / `backbone.last_background_vad_logits`
（interferer 語者的 activity),而不是 foreground 的 label。這個搭配的
head,讓 bottleneck 能有一個明確代表「背景說話者」的表示,但不會讓背景
語音變成 enhanced 輸出的一部分。

因為 `forward` 是原封不動繼承來的,這個 class **本身沒有**任何處理
缺失/`None` target 的機制——它跟 `VADHeadBCELoss` 一樣照樣會丟例外。這份
文件先前的版本,把「合成一個帶著計算圖邊、值為零的 loss」這個機制歸在這個
class 本身,那是錯的,它不是長在這裡。真正的防護機制高了一層,在
`EncDecMaskBase.compute_loss`（`puresound/system/siso.py`)裡:

```python
elif getattr(loss_func, "uses_background_vad_logits", False):
    bg_target = None if batch is None else batch.get("background_vad_target")
    # When no sample in the batch carries background speech, the dataset/
    # collate pipeline emits no `background_vad_target` key at all (it only
    # zero-fills missing rows when *some* row has background speech). An
    # all-silent batch is still a valid signal -- the target is simply
    # all-zeros -- so synthesize it rather than crashing on a None target.
    if bg_target is None and background_vad_logits is not None:
        bg_target = torch.zeros_like(background_vad_logits)
    weighted_loss = weighted * loss_func(background_vad_logits, bg_target)
```

`compute_loss` 會在呼叫這個 loss *之前*,先合成一個全零的**target**——loss
接下來還是照常執行它原本（數值不為零、貨真價實)的 BCE 運算,只是它自己
從來沒真的看過 `None`。這是一個真正修好了某個真實故障的 fix（過去確實有
一整個 batch 都沒有背景語者、target 是 `None`,直接讓 loss crash、整個 DDP
job 死掉的案例);對應的 regression 測試見
`test/test_system/test_siso_compute_loss.py::test_background_vad_loss_handles_all_silent_batch`
與 `::test_background_vad_loss_matches_explicit_zero_target`,後者驗證了
「自動合成出來的路徑」跟「明確傳入零 target」算出來的 loss 完全一致。

還有一點,如果你去追這個機制的話值得知道:目前的 `DPCRN` backbone
（`puresound/nnet/dpcrn.py`)有接 `vad_head` 跟 `dist_head`,但沒有
`background_vad_head`——`backbone.last_background_vad_logits` 是
`compute_loss` 用 `None` 當預設值去讀的一個 hook（`getattr(..., None)`),
不是目前這個隨附的 backbone 會去填的東西。在這個 repo 裡,
`BackgroundVADHeadBCELoss` 目前只被
`egs/voice_isolate/config/exp/backup/` 底下、已經退役的 Conformer-backbone
config 引用。

## Class: `F1_loss`

對二元 frame/utterance 分類問題算的可微分 soft-F1 loss,來自
[asteroid 的 `soft_f1`](https://github.com/asteroid-team/asteroid/blob/fc0967a2eaf42f9446b17f7d039598deffd46f91/asteroid/losses/soft_f1.py)。

```python
F1_loss(eps: float = 1e-10)
```

`forward(estimates, targets) -> Tensor` 直接從 `estimates`/`targets` 算出
soft precision/recall（沒有做 threshold),回傳 `1 - f1.mean()`。目前這個
repo 裡沒有任何 recipe 在用它。
