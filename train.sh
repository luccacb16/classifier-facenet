cmd="python3 train.py \
  --model FaceResNet50 \
  --batch_size 64 \
  --accumulation 512 \
  --epochs 40 \
  --emb_size 512 \
  --min_lr 1e-6 \
  --max_lr 1e-3 \
  --last_epoch -1 \
  --warmup_epochs 4 \
  --num_workers 1 \
  --data_path ./data/ \
  --checkpoint_path checkpoints/ \
  --colab \
  --random_state 42"

echo $cmd
$cmd