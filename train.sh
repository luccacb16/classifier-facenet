cmd="python3 train.py \
  --model arcface \
  --batch_size 32 \
  --accumulation 64 \
  --epochs 40 \
  --emb_size 512 \
  --min_lr 1e-6 \
  --max_lr 1e-3 \
  --warmup_epochs 4 \
  --num_workers 1 \
  --data_path ./data/ \
  --checkpoint_path checkpoints/ \
  --random_state 42"

echo $cmd
$cmd