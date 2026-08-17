# puresound.recipes

English version: [recipes.md](recipes.md)

Config-driven 的建構方式：丟一份 YAML recipe 進去，吐出一個現成的
model / loss list。Model 跟 loss 的型別都是用 `getattr` 在 `puresound.nnet`
與 `puresound.nnet.loss` 上依名稱解析出來的，所以**這兩個 package export
出來的每一樣東西，都能從 config 裡取用**（這個模型庫之所以把所有 backbone
都保持 export 狀態，正是為了這個緣故）。

一份完整、真實、就是建構在這些函式之上的 recipe，可以參考
`egs/voice_isolate/config/train_dpcrn.yaml`——這正是已發佈的 voice-isolate
checkpoint 背後所用的 config。

## 載入 recipe

Config 載入在 [`puresound.config`](configuration.md)，不在這裡。本模組只負責把
一份已驗證的 recipe 所指名的物件建出來。

```python
from puresound.config import load_recipe

recipe = load_recipe(
    path, expected_task="voice_isolation", expected_purpose="train"
)
model = init_siso_model(recipe.model)
loss_list, loss_weights = init_loss_func(recipe.loss_func)
```

`load_siso_recipe_config` 與它那個 20 元素的 tuple 已經移除。那個 tuple 存在的
理由是讓呼叫端可以「解開 config」；typed recipe 自己就帶名字，所以
`recipe.dataset.train_metafile` 取代 index 0，而 `recipe.augmentation_kwargs()`
一次取代 index 6–19。哪些區塊在停用時會變成 `None`，現在由
`BaseRecipe.augmentation_kwargs` 一處決定，而不是看某個 index 當初走的是
`_enabled_config` 還是 `config.get`。

## `init_siso_model(model_dict) -> LightningModule`

建構 `encoder -> features -> backbone`，並把它們包進設定好的 Lightning
module：

```yaml
model:
  lightning_module: {type: EncDecMaskBase, module_args: {mask_type: complex, ...}}
  encoder:          {type: ConvEncDec,     encoder_args: {...}}
  features:         {feats_type: complex, drop_stft_first_bin: True, ...}
  freq_eq:          {type: FrequencyEQLayer, ...}    # optional
  backbone:         {type: DPCRN,          backbone_args: {...}}
```

`type` 字串會分別在 `puresound.system`（lightning module）與
`puresound.nnet`（encoder / freq_eq / backbone）上解析。

## `init_loss_func(hparam_conf) -> (loss_list, weight_list)`

`loss_func` 這個 yaml list 裡的每一項：

```yaml
loss_func:
  - type: SDRLoss          # 在 puresound.nnet.loss 上解析
    weighted: 1.0          # 純量權重
    args: {scaled: False}  # constructor 的 kwargs
```

訓練系統的 loss registry 會同時吃下這兩個 list
（`model.register_loss_func(loss_list, weight_list)`）。
