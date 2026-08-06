# puresound.nnet.loss.spk

English version: [spk.md](spk.md)

Speaker classification 與 verification 用的 loss 函式:兩個 margin-based
softmax 變體,用於 closed-set 分類訓練（`AAMsoftmax`、`SphereFace2`),以及
兩個不需要 classifier head、直接在一個 batch 的 embeddings 上運作的
metric-learning loss（`GE2ELoss`、`TripletLoss`)。

## Class: `AAMsoftmax`

**Additive Angular Margin Softmax（AAM-softmax / ArcFace)**——在做 softmax
之前,先在目標類別的 cosine similarity 上加一個角度 margin,讓訓練直接針對
「speaker 類別之間的角度分離度」最佳化,而不只是分類準確率。

### Constructor

```python
AAMsoftmax(
    embedding_dim: int,
    n_classes: int,
    margin: float = 0.2,
    scale: int = 30,
    mp: float = 0.,
    sub_center: int = 1,
    sub_center_topk: Optional[int] = 0,
    sub_center_type: str = "max",
)
```

**Parameters:**
- `embedding_dim` – 輸入 embedding 的維度。
- `n_classes` – speaker 類別數。
- `margin` – 加在目標類別角度上的角度 margin `m`（單位:弧度)。
- `scale` – 加完 margin 之後套用的 logit scale `s`。
- `mp` – 套用在「難分辨的（confusable)」負類別上的 margin penalty——見下方
  `sub_center_topk`。內部會依 `margin / 0.2` 重新縮放（也就是說 `mp` 是相對
  `margin=0.2` 這個參考點來指定的),若 `margin` 接近 0 則會被歸零。
- `sub_center` – 若 `> 1`,每個類別會有 `sub_center` 個權重向量,而不是只有
  一個（sub-center AAM-softmax),讓一個類別即使 embedding 呈多群聚分佈
  （例如同一個 speaker 在差異很大的錄音條件下錄的樣本),只要樣本命中其中
  *任何一個* sub-center,依然能拿到滿分。
- `sub_center_topk` – 若 `> 0`,對每個樣本最難分辨的前 `k` 個*負*類別
  （跟這個樣本 cosine similarity 最高的非目標類別)套用（比較弱的)`mp`
  margin,而不是讓它們維持原始 cosine——這是疊加在主 margin 之上的
  hard-negative-mining 壓力。
- `sub_center_type` – 把 `sub_center > 1` 收斂成每個類別一個相似度時,選
  `"max"`（取最近的 sub-center)還是 `"avg"`（用 softmax 加權混合各
  sub-center)。

參考文獻:sub-center ArcFace（[Deng et al.](https://ibug.doc.ic.ac.uk/media/uploads/documents/eccv_1445.pdf)),
一個 sub-center 變體（[arXiv:2407.04291](https://arxiv.org/pdf/2407.04291v1)),
以及 top-k hard-negative margin（[arXiv:2110.05042](https://arxiv.org/pdf/2110.05042)）。

### `forward(x, label) -> Tensor`

**只回傳 loss**——不是 `(loss, accuracy)`。`x` 形狀是
`[batch, embedding_dim]`（內部會做 L2-normalize),`label` 形狀是 `[batch]`
（或 `[batch, 1]`,會被 squeeze)。

基本情況（`sub_center=1`、`sub_center_topk=0`)就是標準的 AAM-softmax /
ArcFace cross-entropy:

$$\mathcal{L} = -\log \frac{e^{s \cos(\theta_{y_i} + m)}}{e^{s \cos(\theta_{y_i} + m)} + \sum_{j \neq y_i} e^{s \cos(\theta_j)}}$$

當 `sub_center_topk > 0` 時,最難分辨的前 `k` 個負類別,額外會拿到
`mp`-margin 的 logit（`cos(theta) * cos_mp + sin(theta) * sin_mp`),而不是
單純的 `cos(theta)`,然後才對所有類別取同一個 cross-entropy。

### Config usage

```yaml
# egs/speaker_embedding/conf/PS-spk-v1.yaml / egs/target_speaker_extraction/config/default_config.yaml
loss_func:
  - type: AAMsoftmax
    weighted: 1
    args:
      embedding_dim: 192
      n_classes: 21615
      margin: 0.3
      scale: 30
```

這個 repo 裡的兩個 recipe 用的都是最基本的情況——`mp` / `sub_center` /
`sub_center_topk` 都留在預設值（也就是關閉)。

---

## Class: `SphereFace2`

AAM-softmax 的另一種替代方案:把 speaker 分類重新表述成 `n_classes` 個
獨立的二元（同類別 / 不同類別)判斷,而不是對所有類別做一次 softmax——
這也是參考論文標題「Binary Classification is All You Need」的由來。

### Constructor

```python
SphereFace2(
    in_features,
    out_features,
    scale=32.0,
    margin=0.2,
    lanbuda=0.7,
    t=3,
    margin_type="C",
    sub_center: int = 1,
)
```

**Parameters:**
- `in_features` / `out_features` – embedding 維度 / 類別數。
- `scale` – logit scale（跟 `AAMsoftmax.scale` 角色相同)。
- `margin` – margin;依 class docstring,`margin_type="C"` 建議用 `0.2`,
  `"A"` 建議用 `0.15`（論文的「LMF」設定則分別是 `0.3` / `0.25`)。
- `lanbuda` – positive-pair 項相對於 negative-pair 項（因為有
  `n_classes - 1` 個,數量上占優勢)的權重——沒有這個權重,對所有負類別的
  加總會直接把唯一一個 positive 項淹沒掉。
- `t` – `fun_g(z) = 2 * ((z + 1) / 2)^t - 1` 裡的指數,這是套 margin 之前,
  對 cosine 分數做的一個單調重參數化,用來（依 docstring 說法)「調整分數
  分佈」。
- `margin_type` – `"A"` 把 margin 加在角度裡（ArcFace 風格,
  `cos(theta + margin)`,需要 `sin(theta)`);`"C"` 直接對 cosine 值做
  加減（CosFace 風格,`cos(theta) - margin`)。
- `sub_center` – 跟 `AAMsoftmax` 一樣,但這裡收斂 sub-center 時只實作了
  `"max"`（沒有 `"avg"` 選項)。

`update(margin=0.2)` 這個 method 可以就地重新計算跟 margin 有關的三角函數
常數,不用重建整個 module 就能在不同訓練階段換 margin。

### `forward(input, label) -> Tensor`

**只回傳 loss。** 每個類別都同時算出一個 positive-pair 分數
（`cos_p_theta`,一個把目標類別加了 margin 的 cosine 值往上拉的 softplus
loss)跟一個 negative-pair 分數（`cos_n_theta`,把其他每個類別加了 margin
的 cosine 值往下壓);目標類別貢獻它的 `cos_p_theta`,其他每個類別貢獻它的
`cos_n_theta`,分別用 `lanbuda` / `1 - lanbuda` 加權,加總後對 batch 取
平均。（原始碼裡留著一行註解掉的 `# return output, loss`——這個 class
現在已經不會回傳那個可以拿來算 accuracy 的 `output` 了。）

目前這個 repo 裡沒有任何 recipe 在用它。

---

## Class: `GE2ELoss`

Generalized End-to-End 的 speaker-verification loss（softmax 或 contrast
兩種變體),從
[`cvqluu/GE2E-Loss`](https://github.com/cvqluu/GE2E-Loss/blob/master/ge2e.py)
移植過來。Centroid 的計算會**排除**目前這筆 utterance 本身（這樣一個
speaker 的 centroid 就不會包含正在拿來跟它比對的那個樣本),整個運作在一個
`[nspks, putts, D]` 的 embedding batch 上。

```python
GE2ELoss(
    nspks: int,
    putts: int,
    init_w: float = 10.0,
    init_b: float = -5.0,
    loss_method: str = "softmax",  # "softmax" | "contrast"
    add_norm: bool = True,
)
```

**Parameters:**
- `nspks` / `putts` – 每個 batch 的 speaker 數 / 每個 speaker 的 utterance
  數（`forward` 會用這兩個數字,把輸入的 `[nspks * putts, D]` embeddings
  reshape 成 `[nspks, putts, D]`)。
- `init_w` / `init_b` – 套用在 cosine-similarity 矩陣上、可學習的仿射
  scale/bias 的初始值（`w`、`b` 是 `nn.Parameter`,會跟模型其他部分一起
  訓練)。
- `loss_method` – `"softmax"`（每個 utterance 的 loss,是對所有 speaker
  centroid 相似度做 softmax cross-entropy)或 `"contrast"`（跟最接近的
  不匹配 centroid 做 sigmoid contrast)。
- `add_norm` – 在算 centroid/相似度之前,先對每個 embedding（沿著 feature
  維度)做 L2-normalize。

`forward(dvecs, label=None) -> Tensor` 回傳逐 utterance loss 加總後的結果
（`label` 參數會被接受但沒有用到)。

---

## Class: `TripletLoss`

在 `[N, 3, D]`（anchor、positive、negative)的 embeddings 上算 triplet
loss。

```python
TripletLoss(margin: float = 0.0, add_norm: bool = True, distance: str = "Euclidean")
```

`forward(x, reduction=True)` 把 `x` 沿 dim 1 拆成 anchor/positive/negative,
視需要做 L2-normalize（`add_norm`),再用 `euclidean_distance` 或
`cosine_similarity`（由 `distance` 選擇,不分大小寫;其他字串會丟出
`NameError`)算出 `dist_pos` / `dist_neg`,回傳
`mean(max(0, dist_pos - dist_neg + margin))`（若 `reduction=False` 則回傳
未經 reduce 的逐 row 結果)。
