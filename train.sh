cmd="python3 train.py \
  --batch_size 128 \
  --accumulation 1024 \
  --epochs 30 \
  --emb_size 512 \
  --lr 1e-4 \
  --num_workers 1 \
  --data_path ./data/ \
  --checkpoint_path ./checkpoints/ \
  --colab False \
  --wandb True"

echo $cmd
$cmd