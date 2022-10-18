import os, torch
import sys

from utils.glide_util import generate_raw_image
from utils.text_util import generate_text

# COLORS = ['white', 'beige', 'yellow', 'red', 'orange', 'green', 'blue', 'purple', 'pink', 'black', 'beige']

# from configs import cfg, device_num, total_device
from configs import cfg

#print(cfg)
#print(device_num)
#print(total_device)

Output_path = cfg.Output_path
text_format = cfg.text_format
prefix = cfg.prefix
suffix = cfg.suffix
#print(Output_path)

dataset_name = 'noisy'

NumImages = 8016
batch_size = 4
texts = [text_format] * NumImages

# print(sys.argv[0])
# print(sys.argv[1])
# print(sys.argv[2])
# print(sys.argv[3])
# device_num = int(sys.argv[2]) - 1
# total_device = int(sys.argv[3])

# devices = cfg.device_num
device_num = cfg.device_num - 1
total_device = cfg.total_device

if not os.path.exists(Output_path):
    os.makedirs(Output_path)

# assert NumImages % batch_size == 0
Total_iter = NumImages // batch_size
#print(Total_iter)
#print(total_device)
assert Total_iter % total_device == 0

length = Total_iter // total_device
#print(length)
device = 'cuda:' + str(device_num)

generate_raw_image(texts, length, device=device, batch_size=batch_size, path=os.path.join(Output_path, dataset_name), prefix=prefix, suffix=suffix)

"""
CUDA_VISIBLE_DEVICES=2,3,4 python Image_generator.py 1 1
"""