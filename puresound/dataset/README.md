# puresound.dataset

繁體中文版本：[`README.zh-TW.md`](README.zh-TW.md)

Two manifest styles:

* **Dynamic** (`dynamic_base.py` + `parser.py`, active): a CSV metafile of clean
  utterances; mixtures are synthesized on the fly per item. The task datasets in
  `puresound/task/` build on this.
* **Kaldi-form** (`kaldi_base.py`, legacy): pre-mixed `wav.scp`-style lists,
  used by the frozen SV/TSE recipes.

Full API reference: `docs/dataset/`.
