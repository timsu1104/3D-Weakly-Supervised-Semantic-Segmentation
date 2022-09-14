To train a small U-Net with 5cm-cubed sparse voxels:

1. Link [train](/share/suzhengyuan/data/ScanNetv2/PointGroup/dataset/scannetv2/train) and [val](/share/suzhengyuan/data/ScanNetv2/PointGroup/dataset/scannetv2/val) under this folder.
2. Run 'python prepare_data.py' to generate point cloud data under folder `train_processed` and `val_processed`.
2. Run 'python prepare_text_data.py' to generate text data from json file under folder `train_processed` and `val_processed`.
3. Run 'python unet.py > train.log 2>&1 &' to train the network.

You can train a bigger/more accurate network by changing `m` / `block_reps` / `residual_blocks` / `scale` / `val_reps` in unet.py / data.py, e.g.
```
m=32 # Wider network
block_reps=2 # Deeper network
residual_blocks=True # ResNet style basic blocks
scale=50 # 1/50 m = 2cm voxels
val_reps=3 # Multiple views at test time
batch_size=5 # Fit in 16GB of GPU memory
```
