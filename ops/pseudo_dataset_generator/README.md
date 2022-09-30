### Preparing Mask-pseudo-dataset

First modify the settings in config.py, then run the following commands.

```bash
./create_dataset.sh
python -u Image_filter.py # this will only keep images with white background
python -u extract_mask.py
```

To blur the masks, you will need to have point2mask installed. Then run 
```bash
python -u preprocess_mask.py
```