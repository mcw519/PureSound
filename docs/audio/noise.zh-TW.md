# puresound.audio.noise

English version: [`noise.md`](noise.md)

兩個底層的噪音混音基本函式。[`augmentation.AudioEffectAugmentor`](augmentation.md)
在外面包了一層 pool 管理、重新取樣、跟 id 記錄
(`AudioEffectAugmentor.add_bg_noise`/`add_bg_white_noise`);
`puresound/task/*.py` 在噪音波形已經拿到手的情況下,也會直接呼叫這兩個
module-level 函式。兩者都是**一次呼叫處理一整串目標 SNR**,回傳的也是
**list**,每個 SNR 一筆 —— 沒有「單一 SNR、回傳單一 tensor」的簡化版本。

## Functions

### `add_bg_noise(wav: Tensor, noise: List[Tensor], snr_list: List[float]) -> Tuple[List[Tensor], List[Tensor]]`

把一個或多個噪音波形,以 `snr_list` 裡每一個 SNR 分別混進 `wav`:

```
noisy = wav + scale * noise      # scale 是針對每個 snr_db 解出來,讓 SNR 成立
```

演算法:
1. `noise` 裡的每個 tensor 都會被強制轉成單聲道(如果還不是的話,用
   `n[0].view(1, -1)`),各自做 RMS 正規化,然後全部**沿時間軸串接**成
   一個 noise bed,再整體做一次 RMS 正規化。傳入 `len(noise) > 1` 正是
   `AudioEffectAugmentor.add_bg_noise(..., dynamic_type=True)` 底層在做
   的事 —— 2 段噪音頭尾接成一個 noise bed,而不是 1 段。
2. 如果這個 noise bed 比 `wav` 長,會在隨機偏移處裁切;如果比較短,則會
   重複(`repeat`)後再裁切 —— 輸出永遠精確對齊 `wav` 的長度。
3. 對 `snr_list` 裡的每個 `snr_db`:`bg_rms = rms(wav) / 10**(snr_db / 20)`
   (`rms(wav)` 只在最後一個軸上做 reduce —— 見
   [`volume.calculate_rms`](volume.md),所以多聲道的 `wav` 會得到
   per-channel 的 `bg_rms`,再跟單聲道的 noise bed 做 broadcast);把
   `wav + bg_rms * bed` 加進 noisy-speech list,並把 `bg_rms * bed`
   (那個 SNR 下實際加進去的噪音)加進 noise list。

回傳 `(noisy_speech_list, noise_list)`,**兩者長度都是 `len(snr_list)`**
—— 不是跟 `wav` 同 shape 的 tensor。

---

### `add_bg_white_noise(wav: Tensor, snr_list: List[float]) -> Tuple[List[Tensor], List[Tensor]]`

跟上面一樣是「一串 SNR / list 輸出」的合約,但這裡的「噪音」是每次呼叫
都重新產生的零均值高斯噪音
(`torch.FloatTensor(wav.shape[-1]).normal_(0, std)`,`std` 用跟上面一樣的
方式從 `snr_db` 解出來)—— 沒有 noise pool,也沒有檔案 I/O。當不需要
真正錄製的噪音時,可以拿來當作便宜的噪音來源。

有色(非白)噪音的產生,在原始碼裡列為 `# TODO`,目前尚未實作。

## 範例

```python
from puresound.audio.noise import add_bg_noise, add_bg_white_noise

noisy_list, added_noise_list = add_bg_noise(wav=speech, noise=[noise_wav], snr_list=[0.0, 10.0])
noisy_0db, noisy_10db = noisy_list

white_noisy_list, _ = add_bg_white_noise(wav=speech, snr_list=[5.0])
noisy_5db = white_noisy_list[0]
```
