# puresound.recipes

High-level recipes for constructing models and loss functions from YAML configuration files.

## Functions

### `_enabled_config(config: Dict, key: str) -> Dict`

Internal helper that extracts a sub-configuration block identified by `key`, only if that block is marked as enabled.

**Parameters:**
- `config` – Full configuration dictionary
- `key` – Section key to extract

**Returns:** Sub-configuration dictionary, or empty dict if not enabled.

---

### `load_siso_recipe_config(f_path: str) -> Tuple`

Loads a SISO (Single-Input Single-Output) training configuration from a YAML file.

**Parameters:**
- `f_path` – Path to the YAML config file

**Returns:** `Tuple` of configuration sections:
- `model_dict` – Encoder, feature, and backbone configurations
- `loss_conf` – Loss function configurations
- `optim_conf` – Optimizer/scheduler configuration
- `dataset_conf` – Dataset configuration

---

### `init_siso_model(model_dict: Dict) -> Tuple`

Constructs the encoder–feature–backbone model pipeline from a configuration dictionary.

Expected `model_dict` keys:
- `encoder` – Encoder module config (e.g., `FreeEncDec` or `ConvEncDec`)
- `feature` – Feature processing module config (e.g., `MelBank`)
- `backbone` – Main network backbone config (e.g., `DPRNN`, `UNet`)

**Returns:** `Tuple(encoder, feature_encoder, backbone)`

---

### `init_loss_func(hparam_conf: List) -> List`

Initializes a list of loss functions from configuration, each with an associated weight.

**Parameters:**
- `hparam_conf` – List of loss function configuration dictionaries. Each entry should contain:
  - `name` – Loss function class name
  - `weight` – Scalar weight for this loss term
  - Additional kwargs passed to the loss constructor

**Returns:** List of `(loss_fn, weight)` tuples.

## Example YAML Config (SISO)

```yaml
encoder:
  enabled: true
  type: FreeEncDec
  params:
    win: 16
    stride: 8
    out_channel: 512

feature:
  enabled: false

backbone:
  type: DPRNN
  params:
    in_channel: 512
    hid_channel: 128
    num_layers: 6

loss:
  - name: SDRLoss
    weight: 1.0
    mode: sisnr
```
