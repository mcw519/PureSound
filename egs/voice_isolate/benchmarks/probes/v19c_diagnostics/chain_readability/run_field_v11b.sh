cd /home/milowu/A4Audio/PureSound/egs/voice_isolate
uv run python benchmarks/probes/anchor_gate_cache.py field config/exp/train_dpcrn_v11b_compinv.yaml \
  --ckpt /home/milowu/A4Audio/PureSound/egs/voice_isolate/exp/dpcrn_v11b_compinv/lightning_logs/version_0/checkpoints/epoch=31-step=16000.ckpt \
  --tag v11b --out /tmp/claude-1001/-home-milowu-A4Audio-PureSound/c52f3b49-8e8f-4f9f-a5ca-862771bd494c/scratchpad/agcache --device cuda:1 2>&1 | grep -vE "not in the model|Loaded params"
echo FIELDDONE_V11B
