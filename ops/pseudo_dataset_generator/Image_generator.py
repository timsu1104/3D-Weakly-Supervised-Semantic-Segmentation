import os, torch
import sys

from utils.glide_util import generate_raw_image
from utils.text_util import generate_text

# COLORS = ['white', 'beige', 'yellow', 'red', 'orange', 'green', 'blue', 'purple', 'pink', 'black', 'beige']

from configs import cfg

Output_path = cfg.Output_path
text_format = cfg.text_format

dataset_name = 'noisy'

NumImages = 9000
batch_size = 25
texts = [text_format] * NumImages

device_num = int(sys.argv[1]) - 1
total_device = int(sys.argv[2])

if not os.path.exists(Output_path):
    os.makedirs(Output_path)

assert NumImages % batch_size == 0
Total_iter = NumImages // batch_size
assert Total_iter % total_device == 0
length = Total_iter // total_device
device = 'cuda:' + str(device_num)

generate_raw_image(texts, length, device=device, batch_size=batch_size, path=os.path.join(Output_path, dataset_name), prefix='', suffix='')

"""
CUDA_VISIBLE_DEVICES=2,3,4 python Image_generator.py 1 1
"""