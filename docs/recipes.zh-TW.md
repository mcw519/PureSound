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

## `load_siso_recipe_config(f_path) -> Tuple`

把一份 recipe YAML 解析成一個依位置排列的 20-tuple。呼叫端應該只拆出自己
需要的欄位，並用 `*` 這種寫法去承接未來新增的欄位——這個 tuple 只會用
append 的方式成長：

| index | field | yaml section |
|---|---|---|
| 0 | corpus | `dataset` |
| 1 | trainer | `trainer`（含 `lightning_trainer_args`） |
| 2 | optimizer | `optimizer` |
| 3 | scheduler | `scheduler` |
| 4 | loss | `loss_func`（list） |
| 5 | model | `model`（含 `lightning_module`） |
| 6–13 | augmentation blocks | `augmentation_speech`、`_noise`、`_reverb`、`_speed`、`_ir_response`、`_src`、`_hpf`、`_volume` |
| 14–16 | 其他 blocks | `augmentation_codec`、`_packet_loss`、`_target_absent` |
| 17 | vad label | `vad_label` |
| 18–19 | 真實錄音 blocks | `augmentation_realfar`、`augmentation_realnear`（只有 voice-isolation 會用到） |

被 `used: False`（或整個沒寫）擋掉的 block 會變成 `None`——但**只有表格裡
大部分的欄位是這樣**。精確地說：index 6–11、14–16、18–19 都會經過一個內部
的 `_enabled_config()` 輔助函式，檢查 `item.get("used")`，假值就回傳
`None`。而 index **12、13、17**（`augmentation_hpf`、`augmentation_volume`、
`vad_label`）則是直接用單純的 `config.get(key)` 讀出來——只有當 YAML 裡
根本*沒有這個 key* 時才會是 `None`；如果這個 block 有寫、但內部標成
`used: False`，這三個欄位還是會照樣把整個 block 原封不動回傳（跟其餘
十七個不一樣）。如果你要直接拿這三個欄位其中之一來用，得自己檢查
`used`。

```python
(corpus, trainer, _opt, _sch, _loss, model_dict, *rest) = load_siso_recipe_config(path)
```

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
