# Data Preparation - ScanNetV2

To prepare ScanNetV2 data for training:
1. Link [train](/share/suzhengyuan/data/ScanNetv2/PointGroup/dataset/scannetv2/train) and [val](/share/suzhengyuan/data/ScanNetv2/PointGroup/dataset/scannetv2/val) under this folder. Suppose you are under this folder, then the following command can be used: 
```
ln -s /share/suzhengyuan/data/ScanNetv2/PointGroup/dataset/scannetv2/train ./train
ln -s /share/suzhengyuan/data/ScanNetv2/PointGroup/dataset/scannetv2/val ./val
```
2. Run `python prepare_data.py` to generate point cloud data under folder `train_processed` and `val_processed`.
3. Run `python prepare_text_data.py` to generate text data from json file under folder `train_processed` and `val_processed`.
