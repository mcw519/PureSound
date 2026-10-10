# puresound.audio.room_simulator

English version: [room_simulator.md](room_simulator.md)

`RoomImpulseResponseSimulator` 以 image-source 法（J. B. Allen and D. A. Berkley,
*Image method for efficiently simulating small-room acoustics*, JASA 1979）即時
產生 shoebox 房間的 RIR，底層用 E. Habets 的 `rir_generator`。它是預先產生 bank
（[rir_bank.zh-TW.md](rir_bank.zh-TW.md)）的即時替代方案；recipe 透過
`AudioEffectAugmentor.init_room_simulator` 與 `apply_rir` 使用它
（[augmentation.zh-TW.md](augmentation.zh-TW.md)）。

## 先抽場景，每次呼叫再放聲源

`sample_scene()` 固定房間、接收點與 RT60，但**沒有聲源**。`generate()` 每次呼叫
都放一個新聲源，距離由 `source_role` 決定。因此同一個場景可以同時容納近場前景
說話者、遠場干擾者與貼牆的媒體聲源，全部由同一支麥克風在同一個房間收到：
dataset 對每個聲源各呼叫一次 `apply_rir`，場景相同、角色不同。

## 設定

```python
@dataclass
class RoomImpulseResponseSimulator:
    room_dim_range: list[list[float]]              # [[xmin,xmax],[ymin,ymax],[zmin,zmax]]，m
    rt60_range: list[float]                        # [min, max]，s
    source_receiver_distance_range: list[float]    # [min, max]，m；任何角色的後備
    foreground_distance_range: Optional[list[float]] = None
    interferer_distance_range: Optional[list[float]] = None
    media_distance_range: Optional[list[float]] = None
    receiver_margin: float = 0.4                   # 接收點離牆最小距離，m
    source_margin: float = 0.4                     # 聲源離牆最小距離，m
    media_wall_offset_max: float = 0.2             # 媒體聲源離所貼牆面的最大額外距離，m
    sound_speed: float = 343.0                     # m/s
    nsample: Optional[int] = None                  # RIR 長度；None = int(RT60 · fs)
    order: int = -1                                # 反射階數；-1 = 全部
    hp_filter: bool = True                         # Allen–Berkley 高通（去除累積的 DC）
```

YAML 區塊是 `augmentation_reverb.simulator`，key 相同，另有 `used`、
`source_level`、`pregenerated`，由 dataset 層讀取、`init_room_simulator` 剝除。
取樣率是 `generate()` 的參數，不是設定。

## `sample_scene() -> {"room_dim", "receiver", "rt60"}`

房間尺寸由 `room_dim_range` 逐軸均勻抽；接收點在房內均勻抽，離每面牆至少
`receiver_margin`；RT60 由 `rt60_range` 均勻抽。全部使用 NumPy 的全域 generator。

## `generate(sample_rate, scene=None, source_role="source", distance_range_override=None) -> (rir, metadata)`

1. `scene` 預設為新的 `sample_scene()`。
2. 聲源的**距離範圍**：

   | 條件 | 範圍 |
   | --- | --- |
   | 有給 `distance_range_override` | 就用它，精確遵守 |
   | `source_role="foreground"` | `foreground_distance_range` |
   | `source_role="media"` | `media_distance_range`，否則 `interferer_distance_range` |
   | `source_role="interferer"` | `interferer_distance_range` |
   | 其他值，或該角色的範圍為 `None` | `source_receiver_distance_range` |

3. **擺放。** 最多抽 64 次離牆至少 `source_margin` 的點，與接收點的距離落在範圍內
   就接受。`"media"` 聲源會貼到隨機一面 x 或 y 牆，距離為
   `source_margin + U(0, media_wall_offset_max)`，因為電視或喇叭靠牆擺放，其早期
   反射比同距離、獨立站立的說話者更強。
4. **球殼後備。** 很薄的球殼（例如 `[0.3, 0.5]` m）很少接受房內均勻抽樣，此時直接
   在球殼上抽：均勻半徑與方向，最多 256 次，只以離牆距離拒絕。若球殼與房間沒有
   交集，就把聲源放在朝最遠房內角落的方向，半徑截到該角落的距離，即可達成的最近
   距離。這條路徑不做媒體貼牆：距離標籤優先於擺放。
5. **產生。** 以全指向接收點呼叫 `rir_generator.generate(c, fs, r, s, L,
   reverberation_time=rt60, nsample, order, hp_filter)`。給定 RT60 時，
   `rir_generator` 反推 Sabine 公式，得到六面牆共用、與頻率無關的吸收係數
   `α = 24·ln10·V / (c·S·RT60)`，反射係數 `β = √(1 − α)`；`α > 1`（RT60 對抽到的
   房間太短）時 raise `ValueError`。

回傳 `rir: Tensor[1, T]`（float32）與

```python
{"room_dim": [x, y, z], "receiver": [x, y, z], "source": [x, y, z],
 "rt60": float, "source_role": str,
 "source_receiver_distance": float,           # 實際距離，m
 "drr_db": float}                             # compute_drr_db，2.5 ms 窗
```

## 精確距離：模擬器與 bank 的差別

這裡的 `distance_range_override` 會被精確遵守，因為模擬器是即時產生；預先產生的
bank 只能挑出最接近所要求範圍的已存通道。函式庫中使用 override 的地方：

- `puresound/task/ns.py` 的殘餘播放回音：裝置自己的喇叭，位於
  `augmentation_speech.echo_playback.distance_range`（預設 `[0.2, 1.0]` m）。
  真正的近場回音通道需要模擬器；用 bank 時回音拿到的是最接近的已存通道。
- `puresound/task/session_rows.py`：使用者在 `user_distance_range`、旁人在
  `bystander_distance_range`。

## 設計限制

模型是空的 shoebox，吸收均勻且與頻率無關，聲源與接收點都是全指向點，只有鏡面
反射。沒有家具、沒有散射、沒有頻率相依的衰減，也沒有聲源指向性。這些由 RIR 生成
堆疊補上（見 [hybrid RIR](hybrid_rir.zh-TW.md) 與 [scene schema](rir_scene_v2.zh-TW.md)）。

## 範例

```python
from puresound.audio.room_simulator import RoomImpulseResponseSimulator

sim = RoomImpulseResponseSimulator(
    room_dim_range=[[3, 8], [3, 6], [2.5, 4]],
    rt60_range=[0.15, 0.8],
    source_receiver_distance_range=[0.3, 4.0],
    foreground_distance_range=[0.3, 1.2],
    interferer_distance_range=[1.2, 4.0],
    nsample=8192,
)
scene = sim.sample_scene()
fg_rir, fg_meta = sim.generate(16000, scene=scene, source_role="foreground")
it_rir, it_meta = sim.generate(16000, scene=scene, source_role="interferer")
# 同一房間與接收點；前景距離在 [0.3, 1.2] m，干擾者在 [1.2, 4.0] m
```
