# puresound.task.sampler

English version: [`sampler.md`](sampler.md)

作為 `DataLoader` 的 `batch_sampler` 使用的 **batch** sampler：每次 `__iter__`
yield 出來的就已經是一整個 batch 的 item keys，直接交給 task dataset 的
`__getitem__`（見 [task.ns](ns.zh-TW.md)），後者用同一個 parser
（`DynamicBaseDataset.parse_item_key`）讀所有 key 形狀。

Key 只會隨 run 要求的東西變長：

| key | 何時 |
| --- | --- |
| `(speaker, sr)` | 一律 |
| `(speaker, sr, item_seed)` | sampler 有 seed（deterministic validation） |
| `(speaker, sr, item_seed_or_None, seconds)` | 設了 `length_schedule` |
| `(speaker, sr, item_seed_or_None, seconds_or_None, epoch)` | 開了 `emit_epoch` |
| `(speaker, sr, item_seed, seconds_or_None, epoch_or_None, utterance)` | `CoverageSampler`（由它指定句子） |

空的欄位以 `None` 佔位，所以光看長度就知道是哪一種形狀。除非開了
`select_by_sr_first`，否則 `sr` 為 `None`。什麼都沒要求的 run 拿到的是兩個元素的 key。

## Class: `SpeakerSampler`

每個 batch 含 `n_spks` 位 speaker、每位 `n_per` 個 item。這讓一個 batch（a）有足夠
多不同的 speaker，供干擾者抽樣抽「另一位 speaker」；（b）每位 speaker 有多個 item，
供每步每個類別需要多個樣本的 embedding loss 使用。

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
    length_schedule: Optional[List[Tuple[float, int, float]]] = None,
    emit_epoch: bool = False,
    source_weights: Optional[Dict[str, float]] = None,
)
```

**參數：**
- `data`——dataset 的 `meta` dict（`SpeakerSampler(data=train_dataset.meta, ...)`）：
  以 speaker id 為 key，每個值至少有一個 `"utts"` 子 dict。只讀 speaker id；
  `select_by_sr_first=True` 時另讀 `data[spk]["utts"][utt]["sr"]`。建構完成後
  sampler 會丟掉對 `data` 的參照，所以不會多存一份語料 metadata。
- `total_batch`——每個 epoch 的 batch 數（`__len__`）。
- `n_spks`——每個 batch 的不同 speaker 數。超過 speaker pool 大小時，constructor
  會把它夾成 `n_spks = len(pool)`，並把 `n_per` 提高為
  `ceil(n_spks * n_per / len(pool))` 以維持大致相同的 batch 大小，同時記一則 warning。
- `n_per`——每個 batch 每位 speaker 的 item 數。
- `fast_sampling`——先把 speaker pool 洗牌一次，切成每組 10,000 人（餘數併入最後一組）；每個 batch 先挑
  一組，再從組內挑 `n_spks` 位。這是為超大 pool 做的速度取捨，代價是不再完全均勻。
  迭代時它優先於 `select_by_sr_first`。
- `select_by_sr_first`——每個 batch 先挑一個取樣率，再從該取樣率的 pool 挑
  `n_spks` 位 speaker，所以整個 batch 共用一個 `sr`。Recipe 沒有
  `target_sample_rate` 時 runner 就會開它，因為原生取樣率混雜的 batch 無法堆成一個
  tensor。
- `seed`——啟用 deterministic 模式（見下）。
- `rank` / `world_size`——覆寫從 `torch.distributed` 讀到的值（未初始化時為
  `(0, 1)`）。給測試與想重現某個 rank 串流的呼叫端用。
- `length_schedule`——`(seconds, n_spks, prob)` 分桶。每個 batch 抽一桶，讓模型在
  一次 run 裡受多種上下文長度的監督；該桶的 `n_spks` 取代 constructor 的值，因為
  activation 記憶體跟著長度走。抽樣以 `(seed, epoch, batch index)` 為種子，絕不含
  rank：DDP 下各 rank 步調一致，同一步裡兩個 rank 若列長不同，就會對不同 batch 大小
  平均梯度，並在每次同步時卡住。
- `emit_epoch`——在每個 key 後面附上 epoch index。Recipe 的 `curriculum` 就是這樣
  傳到 worker process 裡的 dataset 副本。沒有排程時是關的，所以沒有排程的 run 維持
  原本的 key 形狀與 RNG stream。
- `source_weights`——`{speaker-id 前綴: 權重}`。每個 batch 位置先依權重抽一個來源，
  再從該來源裡抽 speaker（不放回，除非某來源被抽到的次數超過它的 speaker 數）。沒有
  它時每位 speaker 機率相同，所以某語料在 batch 中的佔比是它的 speaker 佔比，而不是
  時數佔比。每位 speaker 必須恰好符合一個前綴（最長者優先）；沒有符合的 speaker 或
  空的前綴會 raise `ValueError`，與 `select_by_sr_first` 或 `fast_sampling` 同時使用
  也會。

### `set_epoch(epoch)`

由外部告知下一輪是哪個 epoch。在內部自己數輪次，在最關鍵的路徑上是錯的：從 epoch N
續跑的 run 會把計數從 0 重新開始，重播任何依 epoch 而定的排程的開頭。Lightning 會在
每個 epoch 的 iterator 被消耗前，對 `dataloader.batch_sampler.sampler` 呼叫它，續跑
的那一輪也一樣；`sampler` property 回傳 sampler 自己，讓這個呼叫落得到。沒有外部
呼叫時使用內部計數，所以單純的 PyTorch 迴圈不受影響。

### 亂數串流

- **有 seed**：`__iter__` 使用私有的 `random.Random(seed + rank * 1_000_003)`，所以
  用相同 `(seed, rank, world_size)` 重建 sampler 會重現相同的 batches——speaker 選擇、
  `sr`、順序——不受 process 中其他地方動過全域 RNG 的影響，且不同 rank 拿到互相獨立
  的串流。
- **無 seed、DDP**（`world_size > 1`）：每個 rank 一個私有串流，以全域 RNG 抽一次
  再依 rank 偏移。
- **無 seed、單一 process**：使用全域的 `random` module。

### Deterministic validation

Runner（`puresound.system.runner.build_dataloaders`）只對 validation sampler 設
seed，training sampler 則持續探索新的抽樣：

```python
valid_sampler = SpeakerSampler(
    data=valid_dataset.meta,
    total_batch=trainer.valid_iter_per_epoch,
    n_spks=trainer.n_spk_per_batch,
    n_per=trainer.n_utt_per_speaker,
    select_by_sr_first=not corpus.target_sample_rate,
    seed=trainer.valid_seed,          # recipe 預設 1234
)
```

Seeded 模式下，每個 key 帶一個由 `(seed, rank, batch index, speaker slot,
replica index)` 推導出的 `item_seed`。`parse_item_key` 會在做任何事之前用它重新
seed `random`、`numpy` 與 `torch`，所以該 item 的整個即時合成——挑哪段
utterance、房間擺位、SIR、每一次增強的擲硬幣——不論 epoch、run 或 DataLoader worker
如何分配，都能位元級一致地重新產生。因此 validation loss 與 metrics 每個 epoch、每次
run 都在同一批合成樣本上計算。`test/task/test_sampler.py` 釘住
「相同 `(seed, rank)` 重現相同 batches、兩個 rank 的 item seeds 不重疊」。

### `__len__() -> int`

回傳 `total_batch`。

### `__iter__() -> Iterator[List[Tuple]]`

Yield `total_batch` 個 batch，每個是 `n_spks * n_per` 個 key 的 list（夾制之後的
值，或 length schedule 下該桶的 `n_spks`），並洗牌讓不同 speaker 的 item 交錯。

### Recipe 設定

```python
from puresound.task.sampler import SpeakerSampler

train_sampler = SpeakerSampler(
    data=train_dataset.meta,
    total_batch=trainer.train_iter_per_epoch,
    n_spks=trainer.n_spk_per_batch,
    n_per=trainer.n_utt_per_speaker,
    select_by_sr_first=not corpus.target_sample_rate,
    length_schedule=[(b.seconds, b.n_spk, b.prob) for b in trainer.length_schedule]
    if trainer.length_schedule else None,
    emit_epoch=curriculum is not None,
    source_weights=trainer.speaker_source_weights,
)
train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_sampler=train_sampler,   # 是 batch_sampler，不是 sampler
    pin_memory=True,
    num_workers=trainer.num_workers,
    collate_fn=collate_fn,
    worker_init_fn=seed_worker,    # puresound.system.runner.seed_worker
)
```

`puresound.system.runner.build_dataloaders` 是唯一建立它們的地方；每個 recipe
driver（`egs/*/main.py`）都呼叫它。

## Class: `CoverageSampler`

抽樣比例跟 `n_per=1`、有 `source_weights` 的 `SpeakerSampler` 相同——每個位置先依權重
選來源、再選語者、再選句子——但語者和句子是從洗牌後的佇列依序取，不是每次重新抽：
一個來源的每位語者都輪過一次才會重複，一位語者的每句話都輪過一次才會重複。用
`trainer.train_sampler: coverage` 選用；validation 仍是有 seed 的 `SpeakerSampler`。

```python
CoverageSampler(
    data: Dict,
    total_batch: int,
    n_items: int,
    default_seconds: float,
    length_schedule: Optional[List[Tuple[float, int, float]]] = None,
    source_weights: Optional[Dict[str, float]] = None,
    emit_epoch: bool = False,
    seed: int = 0,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
)
```

- `data`——dataset 的 `meta`；讀 `data[spk]["utts"][utt]["length"]` 與 `["sr"]`。
- `n_items`——沒有長度排程時每個 batch 的 item 數。
- `default_seconds`——沒有排程時的列長（`training_length_seconds`）。
- `length_schedule`、`source_weights`、`emit_epoch`、`rank`、`world_size`——同 `SpeakerSampler`。

**列長。** 每個精確長度有自己的佇列，只收長度足夠的句子，避免選中的前景因原始檔
過短而補零。沒有夠長句子的語者（或整個來源）就不參加那種長度，並記一行 log。長度
每個 batch 抽一次，所有 rank 相同；NS、SV、TSE 的目標裁切與干擾對齊都採用該長度。
Augmentation 仍可依任務插入靜音，或在 target-absent 列替換前景。

**Rank。** 每個 rank 都算出整個 batch 的所有位置、只取自己那一份，所以各 rank 分到的是
同一條抽樣序列裡互不重疊的部分。

**位置。** 每一次抽取——長度、來源、佇列順序、逐筆合成 seed——都只由 seed 和在序列中
的位置決定，各自讀一條 PCG64，從不碰全域 RNG。抽樣設定相同時，不管怎麼切成 epoch
和 run，選中的資料與合成 seed 都一樣：

- `checkpoint_state(epochs_done, batches_done=None)`——跑完指定數量的完整 epoch，
  再加上本 epoch 已完成的 batch 後的位置。Callback 計算訓練實際消耗的 batch，
  而不是可能已預取的 sampler 輸出；重建時分塊計算，並快取近期位置。
- `start_from(state, resume=True)`——在 epoch 邊界接續同一階段。World size、batch
  數、長度／來源機率、epoch key 設定，以及合格佇列的內容與順序都須符合 checkpoint。
- `start_from(state, resume=False)`——從保存的位置開始新的一階，也支援 `max_steps`
  在 epoch 中途停止的 checkpoint。可改 batch 數、機率與 world size；共用列長的
  佇列仍須包含相同的合格語者、句子與順序，新列長則建立新的佇列。

兩者都保留記錄的 seed。Checkpoint 保存佇列與階段設定的簽章，不相容時會報錯，
避免默默重播或跳過資料。Epoch 中途的 checkpoint 必須用 warm start；Lightning 的
非 stateful dataloader 無法可靠地從 epoch 中途 resume。Callback 也會在訓練開始前
核對 Lightning 恢復的 epoch；若以 `max_steps` 為基礎的迴圈恢復到前一個 epoch，
須改用以 epoch 為基礎的訓練或 warm start。

`puresound.system.sampling` 把狀態寫進每個 checkpoint（`checkpoint["coverage_sampler"]`），
並在 `--ckpt_path` 與 `--pretrained_ckpt_path` 時讀回；兩者都沒有的 run 以 `--set_seed`
開始新的序列。需要 `n_utt_per_speaker: 1` 且設了 `target_sample_rate`。指定句子的音訊若
為空、靜音或含有非有限值，載入會報錯，不會改抽佇列外的另一句。

## Class: `SpeakerGenderSampler`

另一個較簡單的 batch sampler，產生性別平衡的 batch：每個 batch 從 `spk_list_male`
抽 `n_spks / 2` 位、從 `spk_list_female` 抽 `n_spks / 2` 位（有給 `spk_list_other`
時——給缺性別 metadata 的 speaker——則三個 list 各抽 `n_spks / 3` 位），再把每個抽到
的 speaker id 重複 `n_per` 次。`n_spks` 必須能被 2 或 3 組整除（在 `__init__` 中
assert）。`__len__` 是 `total_batch`。

```python
SpeakerGenderSampler(
    total_batch: int,
    n_spks: int,
    n_per: int,
    spk_list_male: List,
    spk_list_female: List,
    spk_list_other: Optional[List] = None,
)
```

它的 batch 是單純的 speaker id，不是 dynamic datasets 解析的 `(speaker, sr, ...)`
key，也沒有 seeded 或 DDP 模式；repository 裡沒有任何地方呼叫它。它被保留為 library
資產，搭配 [`puresound.dataset.corpus.kaldi`](../../usage/data_preparation.zh-TW.md)
的性別 metadata（`utt2gender_path`）使用。
