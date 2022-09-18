# Training Procedure
To train a small U-Net with 5cm-cubed sparse voxels:
1. Follow the instructions to prepare data. 
2. Run 'CUDA_VISIBLE_DEVICES=$SELECTED_DEVICE$ python -u train.py > $LOGFILENAME$.log 2>&1 &' to train the network.

You can train a bigger/more accurate network by changing `m` / `block_reps` / `residual_blocks` / `scale` / `val_reps` in config, e.g.
```
m: 32 # Wider network
block_reps: 2 # Deeper network
residual_blocks: True # ResNet style basic blocks
scale: 50 # 1/50 m :  2cm voxels
val_reps: 3 # Multiple views at test time
batch_size: 5 # Fit in 16GB of GPU memory
```

# Switch Backbone
Register your module under ```models/```. Then switch the backbone name in the config. 

Note that the input should be [coords, feats] and the output should be logits for every points. For details, please refer to the docstring of ```models/SparseConvNet.py```. 