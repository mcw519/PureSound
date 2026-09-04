cd /home/milowu/A4Audio/PureSound/egs/voice_isolate
OUT=/tmp/claude-1001/-home-milowu-A4Audio-PureSound/c52f3b49-8e8f-4f9f-a5ca-862771bd494c/scratchpad/v19_plan/chain_readability
P=$OUT/comp_readability.py
uv run python $P --tag v8 --config config/train_dpcrn.yaml --ckpt pretrained_ckpt/dpcrn_v8.ckpt --device cuda:1 --chain both --out $OUT/comp_v8.json 2>&1 | grep -vE "not in the model|Loaded params"
uv run python $P --tag v16 --config config/exp/train_dpcrn_v16_lengthmix.yaml --ckpt /work/any_exp_link/puresound_exp/dpcrn_v16_lengthmix/lightning_logs/version_0/checkpoints/epoch=19-step=10000.ckpt --device cuda:1 --chain both --out $OUT/comp_v16.json 2>&1 | grep -vE "not in the model|Loaded params"
uv run python $P --tag v11b --config config/exp/train_dpcrn_v11b_compinv.yaml --ckpt exp/dpcrn_v11b_compinv/lightning_logs/version_0/checkpoints/epoch=31-step=16000.ckpt --device cuda:1 --chain both --out $OUT/comp_v11b.json 2>&1 | grep -vE "not in the model|Loaded params"
echo COMPDONE
