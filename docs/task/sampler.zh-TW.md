# puresound.task.sampler

English version: [`sampler.md`](sampler.md)

以 `torch.utils.data.Sampler` 風格實作的**批次（batch）**sampler，作為
`DataLoader` 的 `batch_sampler` 使用：每一次 `__iter__` 產出的就已經是一整個
batch 份量的 `(speaker, sr[, item_seed])` keys，直接交給 task dataset 的
`__getitem__`（見 [task.ns](ns.zh-TW.md)、[task.tse](tse.zh-TW.md)、
[task.sv](sv.zh-TW.md)）。

## Class: `SpeakerSampler`

每個 batch 都包含 `n_spks` 位 speaker，each 有 `n_per` 段 utterance，這正是讓
一個 batch 能夠：(a) 提供足夠多不同的 speaker，給
`task.ns`/`task.tse`/`task.voice_isolation` 裡「從這個 batch 的 speaker pool
中抽某個其他 speaker」的干擾者抽樣邏輯使用；(b) 讓每個 class 在同一步
（step）裡有多筆 sample，供需要這種結構的 embedding loss 使用（GE2E 風格的
batch、`AAMsoftmax`；見 `egs/speaker_embedding/conf/PS-spk-v1.yaml` 裡的
`n_spk_per_batch: 64, n_utt_per_speaker: 2`）。

### Constructor

```python
SpeakerSampler(
    data: Dict,
    total_batch: int,
    n_spks: int,
    n_per: int,
    fast_sampling: bool = False,
    select_by_sr_first: bool = False,
    seed: Optional[int] = None,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
)
```

**參數說明：**
- `data` —— `DynamicBaseDataset` 系列 subclass 從 metafile 建出的 `meta`
  dict（實務上一律是 `dataset.meta`，例如
  `SpeakerSampler(data=train_dataset.meta, ...)`）：以 speaker id 為 key，
  每個 value 至少是一個帶有 `"utts"` 子 dict 的 dict（utterance id ->
  utterance info）。只有 `select_by_sr_first=True` 時才會進一步讀取
  `data[spk]["utts"][utt]["sr"]` 來建立依取樣率分組的 speaker 索引；否則只會
  讀取最上層的 key（speaker id）。Constructor 在抽取完需要的資訊後會立刻捨棄
  對 `data` 的參照（`del self.data`），所以它不會在 dataset 之外再多持有一份
  語料 metadata 的複本。
- `total_batch` —— 每個 epoch `__iter__` 會產出的 batch 數（`__len__`）。
- `n_spks` —— 每個 batch 裡不同 speaker 的數目。若 `n_spks` 超過 speaker
  pool 的大小，constructor **不會** raise：它會把 `n_spks` 夾（clamp）到
  `len(spk_pool)`，並把 `n_per` 依比例放大成
  `ceil(n_spks * n_per / len(spk_pool))`，盡量維持原本要求的 batch 大小，並
  印出一則訊息。
- `n_per` —— 每個 speaker 在每個 batch 裡的 utterance 數（會受上述夾值影響）。
- `fast_sampling` —— 若為 `True`，先把 speaker pool 整體 shuffle 一次，再切
  成每組一萬人的群組；每個 batch 先抽一個群組，再從群組內抽 `n_spks` 位
  speaker，而不是每個 batch 都對整個 pool 做一次 `random.sample`。這是為了
  在極大型 speaker pool 上換取速度、犧牲一點抽樣的均勻度。**在 iteration
  時，這個選項的優先權高於 `select_by_sr_first`**——若兩者同時為
  `True`，SR 分組的路徑會被靜靜地永遠跳過，每次產出的 `sr` 都會維持
  `None`。
- `select_by_sr_first` —— 若為 `True`，每個 batch 會先（在 `data` 中出現過
  的取樣率裡）均勻抽出一個取樣率，再只從該取樣率底下的 speaker pool 抽
  `n_spks` 位 speaker，確保同一個 batch 裡每個 item 共用同一個 `sr`。當
  dataset 沒有強制單一 `target_sample_rate` 時，recipe 就會開啟這個選項
  （`select_by_sr_first=False if corpus_dict["target_sample_rate"] else
  True`），因為原生取樣率不一致的 batch 沒辦法疊成同一個 tensor。
- `seed` —— 開啟**deterministic validation**（見下文）。
- `rank` / `world_size` —— 覆寫原本會從 `torch.distributed` 讀到的 DDP
  rank/world size（若 distributed 尚未初始化，會退回 `(0, 1)`）。主要用於
  測試，或是想重現特定 rank 的 seeded 串流、但本身不在 DDP 環境下的呼叫端。

### Deterministic validation（seeded 模式）

Recipe 只會對 **validation** 的 sampler 設 seed，training 的 sampler 則維持不
設 seed，讓它持續探索新的抽樣結果：

```python
valid_sampler = SpeakerSampler(
    data=valid_dataset.meta,
    total_batch=trainer_dict["valid_iter_per_epoch"],
    n_spks=trainer_dict["n_spk_per_batch"],
    n_per=trainer_dict["n_utt_per_speaker"],
    select_by_sr_first=False if corpus_dict["target_sample_rate"] else True,
    seed=trainer_dict.get("valid_seed", 1234),
)
```

當設了 `seed` 之後：

- **每個 rank 各自獨立的 RNG。** `__iter__` 會建立自己的
  `random.Random(seed + rank * 1_000_003)`，而不是從共用的 `random` module
  抽樣，所以只要用同一組 `(seed, rank, world_size)` 重新建立
  `SpeakerSampler`，每次都會重現一模一樣的 batch 序列（speaker 選擇、
  `sr`、batch 內 shuffle 後的順序），不受 process 中其他地方對全域 RNG 的
  任何影響。不同 rank 會拿到彼此獨立、互不重疊的串流（絕不會抽到同樣的
  item）。
- **Per-item seed。** 每一筆產出的 tuple 會多一個第三元素
  `item_seed`，由 `(seed, rank, batch_idx, speaker slot, replica index)`
  決定性地推導而來。Task dataset 的 `__getitem__`（`task.ns`、
  `task.voice_isolation`）偵測到 3-tuple 形式時，會在做任何事之前先用
  `item_seed` 重新 seed `random`、`numpy`、`torch` 的全域 RNG，於是那個
  item 整條 on-the-fly synthesis chain——抽到哪段 utterance、房間怎麼擺、
  SIR、每一次 augmentation 的擲硬幣結果——不論在哪個 epoch、哪次 run、
  DataLoader worker 怎麼分配，都會重新產生位元級一致的結果。

最終效果：validation 的 loss/metrics 每個 epoch、每次 run 都是在同一批
（literal 相同的）synthetic 範例上算出來的，所以 epoch 之間、run 之間的比較
不會被抽樣雜訊干擾。這件事直接由
`test/test_utils/test_voice_isolate_recipe.py::test_speaker_sampler_uses_distinct_seeded_streams_per_rank`
驗證，它斷言同一組 `(seed, rank)` 會重現一模一樣的 batch，且兩個 rank 的
item seed 彼此絕不重疊。

### `__len__() -> int`

回傳 `total_batch`。

### `__iter__() -> Iterator[List[Tuple]]`

產出 `total_batch` 個 batch。每個 batch 是一個 `List`，元素為
`Tuple[str, Optional[int]]`（seeded 模式下則是
`Tuple[str, Optional[int], int]`），長度為 `n_spks * n_per`（若前面因為
request 過大而被夾過值，則以夾過的數字為準）——內容是
`(speaker_id, sr)` 或 `(speaker_id, sr, item_seed)`——並且經過 shuffle，讓
不同 speaker 的 item 交錯排列，而不是依 speaker 分組。

### Recipe wiring

```python
from puresound.task.sampler import SpeakerSampler

train_sampler = SpeakerSampler(
    data=train_dataset.meta,
    total_batch=trainer_dict["train_iter_per_epoch"],
    n_spks=trainer_dict["n_spk_per_batch"],
    n_per=trainer_dict["n_utt_per_speaker"],
    select_by_sr_first=False if corpus_dict["target_sample_rate"] else True,
)
train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_sampler=train_sampler,   # note: batch_sampler, not sampler
    pin_memory=True,
    num_workers=trainer_dict["num_workers"],
    collate_fn=collate_fn,
)
```

三個會建立 `DynamicBaseDataset` subclass 的 egs main 都是這樣使用它（當成
`batch_sampler=`）：`egs/noise_suppression/main.py`、
`egs/target_speaker_extraction/main.py`、`egs/speaker_embedding/main.py`。

## Class: `SpeakerGenderSampler`

另一個較簡單、獨立的 batch sampler，從固定的男／女（／其他）speaker-id
list 中，各取相等的一半（若有給「其他」list 則是三等分）組成每個
batch——`n_spks` 必須能被 2 或 3 整除。它沒有實作上述的 seeded/DDP 機制，目前
在整個 repo 裡也沒有任何呼叫端在使用它。它究竟會保留、變更、還是被移除，是
另外一項獨立在評估的事；這份文件僅描述它目前的行為。
