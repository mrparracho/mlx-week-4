

nohup uv run python -m src.train_self_attn --self-attn --batch-size 32 --num-epochs 5  --wandb --wandb-run-name "self-attn" > training.log 2>&1 &
nohup uv run python -m src.train_self_attn --self-attn --batch-size 32 --num-epochs 5 --learning-rate 1e-3 --wandb --wandb-run-name "self-attn-lr-1e-3" > training.log 2>&1 &
nohup uv run python -m src.train_self_attn --self-attn --batch-size 64 --num-epochs 5 --learning-rate 1e-4 --wandb --wandb-run-name "self-attn-batch-64" > training.log 2>&1 &


nohup uv run python -m src.train_self_attn --batch-size 32 --num-epochs 5  --wandb --wandb-run-name "cross-attn" > training.log 2>&1 &