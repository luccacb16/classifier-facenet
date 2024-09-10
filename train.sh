cmd="python3 train.py \
  --model FaceResNet50 \
  --batch_size 256 \
  --accumulation 1024 \
  --epochs 30 \
  --emb_size 512 \
  --min_lr 1e-5 \
  --max_lr 3e-4 \
  --num_workers 1 \
  --data_path ./data/ \
  --dataset CASIA \
  --checkpoint_path ./checkpoints/ \
  --colab False"

echo $cmd
$cmd