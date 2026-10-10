# 從 recipe 建出模型與 loss

English: [recipes.md](recipes.md)

`puresound.recipes` 把一份已驗證 recipe 的 `model` 與 `loss_func` 區段變成物件。模型與
loss 的型別都是用 `getattr` 在 `puresound.nnet`、`puresound.system` 與
`puresound.nnet.loss` 上依名稱解析，所以**這些 package export 出來的每一樣東西，都能從
config 取用**——這也是模型庫把每個 backbone 都保持 export 的原因。

建構在這些函式之上的完整 recipe，請看已發布的血統：
`egs/voice_isolate/config/train_dpcrn.yaml` 與
`egs/noise_suppression/config/train_dpcrn_mamba_*.yaml`。

## 從 recipe 到物件

Config 載入在 [`puresound.config`](configuration.zh-TW.md)，不在這裡。typed recipe 自己
就帶欄位名稱（`recipe.dataset.train_metafile`、`recipe.augmentation_kwargs()`），本模組只
負責建出它指名的東西：

```python
from puresound.config import load_recipe
from puresound.recipes import init_loss_func, init_model_for_task

recipe = load_recipe(
    path, expected_task="voice_isolation", expected_purpose="train"
)
model = init_model_for_task(recipe.task)(recipe.model)
loss_list, loss_weights = init_loss_func(recipe.loss_func)
model.register_loss_func(loss_list, loss_weights)
```

`init_model_for_task(task)` 回傳該任務需要的模型工廠：噪音抑制、人聲隔離與語者嵌入用
`init_siso_model`，目標語者擷取用 `init_miso_model`。共用的訓練驅動程式
（`puresound.system.runner`）做的正是這件事。

## `init_siso_model(model_dict) -> LightningModule`

建構 `encoder -> features -> backbone`，並把它們包進設定好的 Lightning module：

```yaml
model:
  lightning_module: {type: EncDecMaskBase, module_args: {mask_type: complex, ...}}
  encoder:          {type: ConvEncDec,     encoder_args: {...}}
  features:         {feats_type: complex, drop_stft_first_bin: True, ...}
  freq_eq:          {type: FrequencyEQLayer, eq_args: {...}}    # 選用
  backbone:         {type: DPCRN,          backbone_args: {...}}
```

`lightning_module.type` 在 `puresound.system` 上解析；`encoder`、`freq_eq` 與
`backbone` 的型別在 `puresound.nnet` 上解析。`features` 是 `nnet.FeatureEncoder` 的
keyword 參數；有 `freq_eq` 區塊時會先建好再傳給它。

`init_miso_model` 吃同樣的形狀，另加條件分支的 `c_encoder` / `c_features` /
`c_backbone`（`module_args.siamese_encoder` 重用混音的 encoder 與 features 時則省略）。

## `init_loss_func(loss_configs) -> (loss_list, weight_list)`

`loss_func` 清單裡的每一項：

```yaml
loss_func:
  - type: SDRLoss          # 在 puresound.nnet.loss 上解析
    weighted: 1.0          # 純量權重
    args: {scaled: False}  # 建構子的 kwargs
```

訓練系統的 loss registry 會同時吃下這兩個 list
（`model.register_loss_func(loss_list, weight_list)`）。`curriculum` 的
`loss:<Type>` 或 `loss:#<index>` track 會依 epoch 移動其中一項的權重（見
[configuration.zh-TW.md](configuration.zh-TW.md#curriculum)）。
