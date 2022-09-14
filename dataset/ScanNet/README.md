To prepare ScanNet data for training:
1. Link [train](/share/suzhengyuan/data/ScanNetv2/PointGroup/dataset/scannetv2/train) and [val](/share/suzhengyuan/data/ScanNetv2/PointGroup/dataset/scannetv2/val) under this folder.
2. Run 'python prepare_data.py' to generate point cloud data under folder `train_processed` and `val_processed`.
2. Run 'python prepare_text_data.py' to generate text data from json file under folder `train_processed` and `val_processed`.
