To train a small U-Net with 5cm-cubed sparse voxels:
1. Follow the instructions to prepare data. 
2. Run 'python train.py > train_scene_level_with_text_cont1.log 2>&1 &' to train the network.

You can train a bigger/more accurate network by changing `m` / `block_reps` / `residual_blocks` / `scale` / `val_reps` in config, e.g.
```
m: 32 # Wider network
block_reps: 2 # Deeper network
residual_blocks: True # ResNet style basic blocks
scale: 50 # 1/50 m :  2cm voxels
val_reps: 3 # Multiple views at test time
batch_size: 5 # Fit in 16GB of GPU memory
```
