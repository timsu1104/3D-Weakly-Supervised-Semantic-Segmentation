# Installation
```bash
pip install -e ops/point2mask 
```

# Training Procedure

1. Follow the instructions under selected dataset (currently only ScanNetV2 is supported) to prepare data. 
2. Run `CUDA_VISIBLE_DEVICES=$SELECTED_DEVICE$ python -u train.py  --config config/$CONFIG_NAME$ > $LOGFILENAME$.log 2>&1 &` to train the network.
3. After training, you can first run `python -u statistics.py` to find out the appropriate logits threshold, and then run `python -u pseudoLabelGeneration.py  --config config/$CONFIG_NAME$`to generate pseudo labels. 
4. After that, prepare a new config specifying the path of pseudo label. See the existing config for reference. 

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
The followings are backbones available now.
- Point cloud encoder 
   1. SparseConvUNet
   2. SparseConvFCNet
   3. SparseConvFCNetEncoder
- Text encoder 
   1. TextTransformer
- Architecture 
   1. MultiLabelContrastive
   2. MultiLabel

If you want to add new module, please register your module under `models/` and specity `embed_length` if it is a point cloud encoder. Then switch the backbone's name in the config. 

Note that the input should be `[coords, feats]` and the output should be logits for every points. If you are using a **SparseConvNet-based point cloud encoder**, just inherit from the class `SparseConvBase_` and **redefine the structure** under `getEncoder(*args)`. The rest will be handled automatically. 