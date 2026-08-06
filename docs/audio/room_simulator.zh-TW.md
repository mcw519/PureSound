# puresound.audio.room_simulator

English version: [`room_simulator.md`](room_simulator.md)

用 image-source method 即時模擬 shoebox 房間 RIR(`rir_generator`,
Habets 對 Allen & Berkley 演算法的實作)。透過
[`augmentation.AudioEffectAugmentor.init_room_simulator`/`.apply_rir`](augmentation.md)
接上;另一種預先產生的替代方案(較快、預先驗證過、沒有 per-item 的
CPU 成本)則是 [`rir_bank.md`](rir_bank.md) 的
`PreGeneratedRoomBank`/`PreGeneratedReleaseBank`。

核心設計概念:**`sample_scene()` 只固定房間跟麥克風位置** —— 這時候還
沒有 source 位置。`generate()` 會在*每一次呼叫*各自取樣 source 位置,
用 `source_role` 參數控制它跟收音端的距離。這就是為什麼一個訓練管線
可以先抽出一個房間,再把 foreground 講者放近、interferer 放遠、把
media/電視類的來源貼著牆放 —— 全部都在*同一個*房間/收音端裡 —— 做法
是用同一個 `scene` dict、但每次帶不同的 `source_role`,重複呼叫
`generate()`(透過 `apply_rir` 間接呼叫)。

## Class: `RoomImpulseResponseSimulator`(dataclass)

```python
@dataclass
class RoomImpulseResponseSimulator:
    room_dim_range: list[list[float]]                        # [[xmin,xmax],[ymin,ymax],[zmin,zmax]] 公尺
    rt60_range: list[float]                                  # [min, max] 秒
    source_receiver_distance_range: list[float]              # [min, max] 公尺 —— 沒有 role 時的預設值
    foreground_distance_range: Optional[list[float]] = None  # source_role="foreground" 時使用
    interferer_distance_range: Optional[list[float]] = None  # source_role="interferer" 時使用
    media_distance_range: Optional[list[float]] = None        # source_role="media";否則退回 interferer_distance_range,再不行就退回基本範圍
    receiver_margin: float = 0.4                              # 麥克風離牆的最小間距,公尺
    source_margin: float = 0.4                                # source 離牆的最小間距,公尺
    media_wall_offset_max: float = 0.2                        # media source 被貼到牆上時,離牆的最大距離
    sound_speed: float = 343.0
    nsample: Optional[int] = None                              # RIR 長度(取樣點數);None = 用 rir_generator 自己的預設值
    order: int = -1                                            # 反射階數;-1 = rir_generator 的預設值(完整 image method)
    hp_filter: bool = True                                      # rir_generator 內建的高通(模擬麥克風膜片的低頻衰減)
```

注意這裡**沒有 `sample_rate` 欄位** —— 取樣率是 `generate()` 呼叫時的
參數,不是模擬器本身的設定,這樣同一個模擬器 instance(以及同一個抽出來
的房間)才能依照當下 recipe 的需求,用任意取樣率去渲染。

### `sample_scene() -> dict`

```python
{"room_dim": np.ndarray[3], "receiver": np.ndarray[3], "rt60": float}
```

從 `room_dim_range` 均勻抽出房間的 x/y/z、在房間內均勻抽一個離每面牆都
至少 `receiver_margin` 的收音點、從 `rt60_range` 均勻抽出 RT60。**沒有
source 位置** —— 那是下面 `generate()` 每次呼叫自己的工作。

### `generate(sample_rate: int, scene: Optional[dict] = None, source_role: str = "source", distance_range_override: Optional[list[float]] = None) -> Tuple[Tensor, dict]`

1. 如果沒給 `scene`,預設會現抽一個新的 `sample_scene()`。
2. 透過內部的 `_sample_source` 抽出一個 source 位置,它的距離範圍是由
   `_distance_range_for_role` 解出來的:

   | `source_role` | 使用的距離範圍 |
   |---|---|
   | 任意值,但有給 `distance_range_override` | 就是 `distance_range_override`,精確使用 |
   | `"foreground"` | `foreground_distance_range`(若為 `None` 則往下一列退) |
   | `"media"` | `media_distance_range`,不行就 `interferer_distance_range`,再不行就退到下一列 |
   | `"interferer"` | `interferer_distance_range`(若為 `None` 則往下退) |
   | 其他任何值(包含預設值 `"source"`) | `source_receiver_distance_range` |

   `"media"` 來源還會額外做**貼牆處理**:抽出來的點,其 x 或 y 座標會被
   夾到隨機選定的某面牆的 `[source_margin, source_margin +
   media_wall_offset_max]` 範圍內 —— 模擬電視/喇叭是貼牆擺放、不是
   自由站立,這會讓它在同樣距離下,早期反射比一般講者更強(依照原始碼
   註解的說法)。
3. 在房間內,針對「離收音端為目標距離」這個條件做 rejection sampling,
   最多嘗試 64 次。如果失敗了 —— 對於很薄的距離殼層(例如 `[0.3,
   0.5]`)這是常態,因為在房間內均勻抽樣很少剛好落在範圍內 —— 就會退回
   `_sample_source_in_shell`:直接在那個距離殼層上取樣(均勻半徑 ×
   均勻方向,最多嘗試 256 次),只用房間邊界來 reject。如果連這個殼層
   都跟房間完全沒有交集,就朝房間內最遠的角落走,把半徑夾到房間幾何
   實際能達到的範圍 —— 也就是用「幾何上能達到的最近距離」去換掉那個
   達不到的精確距離,而不是默默忽略要求的範圍、卻錯貼距離標籤。這個
   殼層 fallback 也會放棄 media 的貼牆處理:當兩者衝突時,距離的正確性
   優先於貼牆位置。
4. 呼叫 `rir_generator.generate(c=sound_speed, fs=sample_rate, r=receiver,
   s=source, L=room_dim, reverberation_time=rt60, nsample=nsample,
   order=order, hp_filter=hp_filter)`,再把結果轉置成 `[channels, T]`
   —— 這裡是 1 個聲道,因為 `r`/`s` 都是單一點,不是麥克風/音源陣列。

回傳 `(rir: Tensor[1, T], metadata: dict)`:

```python
{
    "room_dim": [x, y, z], "receiver": [x, y, z], "source": [x, y, z],
    "rt60": float, "source_role": str,
    "source_receiver_distance": float,
    "drr_db": float,   # 透過 impulse_response.compute_drr_db 計算
}
```

### `distance_range_override`:在這裡是精確的,在預先產生的 bank 裡只是近似值

傳入 `distance_range_override` 會讓模擬器**精確地**在那個範圍內做
rejection sampling,不管設定的 per-role 範圍是什麼。這個 repo 裡實際
用到的地方:

- `puresound/task/ns.py` 的殘留播放回音來源:一個非常近的
  `[0.2, 1.0] 公尺`「通道」,模擬上游 AEC 會留下的喇叭到麥克風洩漏,
  跟 recipe 給一般 interferer 用的 `interferer_distance_range` 完全
  無關(該處的註解寫著:「真正的近場回音通道需要用即時模擬器」)。
- `egs/voice_isolate/scripts/eval_domain_gap.py` 強制指定一個很窄的
  探測距離(`[distance_m - tol, distance_m + tol]`),用來在一個特定、
  受控的距離上評估行為。

**預先產生的 bank 沒辦法精確遵守這個範圍** —— 它只能從池子裡挑一個
*最接近*要求範圍的預先渲染通道,因為它不是即時渲染 RIR(見
[rir_bank.md](rir_bank.md))。真正的近場 override 需要用這個即時模擬器,
不能用 bank。

## Module-level 輔助函式

- **`_sample_range(bounds: list[float]) -> float`** ——
  `Uniform(bounds[0], bounds[1])`。
- **`_sample_room_dim(room_dim_range: list[list[float]]) -> np.ndarray[3]`**
  —— 每個軸各抽一次 `_sample_range`。
- **`_sample_point(room_dim: np.ndarray, margin: float) -> np.ndarray[3]`**
  —— 一個離每面牆都至少 `margin` 的均勻點(對小於 `2*margin` 的房間有
  保護:把取樣範圍下限抬高,確保至少有 1 公分寬)。`margin` 沒有預設值
  —— 每個呼叫點都會明確傳入 `receiver_margin` 或 `source_margin`。

以上這些,加上前面提到的 `_distance_range_for_role`/`_sample_source`/
`_sample_source_in_shell`,都是 private(開頭底線)—— 這裡之所以還是
寫出來,是因為它們*就是* `generate()` 的演算法本身,而不是另外一組
公開介面。

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
fg_rir, fg_meta = sim.generate(sample_rate=16000, scene=scene, source_role="foreground")
it_rir, it_meta = sim.generate(sample_rate=16000, scene=scene, source_role="interferer")
# fg_meta["room_dim"] == it_meta["room_dim"]  (同一個房間、同一個 scene)
# fg_meta["source_receiver_distance"] <= 1.2,it_meta["source_receiver_distance"] >= 1.2
```

實務上通常是透過 `AudioEffectAugmentor.init_room_simulator` +
`.sample_room_scene()` + `.apply_rir(..., room_scene=scene,
source_role=...)`(見 [augmentation.md](augmentation.md))間接使用,
而不是直接建立 instance —— 上面的範例示範的是 `apply_rir` 包裝起來的
底層機制。
