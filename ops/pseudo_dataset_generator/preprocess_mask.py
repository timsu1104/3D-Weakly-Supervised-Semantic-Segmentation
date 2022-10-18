from PIL import Image
import numpy as np, torch
import os, glob, shutil
from tqdm import tqdm, trange
from tqdm.contrib import tzip
import sys
sys.path.append('/home/zhuhe/3DNetWithText_v10.9/3DUNetWithText/ops/')
from point2mask.point2mask_modules import Pixel2Mask
from time import time
from configs import cfg

Output_path = cfg.Output_path
Radius = cfg.blur_radius
Nsample = cfg.blur_samples

input_dataset = os.path.join(Output_path, "mask")
output_dataset = os.path.join(Output_path, f"processed_mask_r{Radius}_nsample{Nsample}")
if os.path.exists(output_dataset):
    shutil.rmtree(output_dataset)
os.mkdir(output_dataset)

Batch_size = 64

files = glob.glob(os.path.join(input_dataset, "*.jpg"))

images = np.stack([np.array(Image.open(fn)) for fn in tqdm(files)])
images[images < 127] = 0
images[images >= 128] = 255
images = torch.from_numpy(images).cuda()
start = time()
masks = []
for i in trange(len(images) // Batch_size + 1):
    batch = images[i * Batch_size: min((i+1) * Batch_size, len(images))]
    masks.append(Pixel2Mask(radius=Radius, nsample=Nsample)(batch, 64))
masks = torch.cat(masks, 0)
filter_num = 0
for fn, mask in tzip(files, masks):
    f = fn.split("/")[-1]
    img = Image.fromarray(mask.cpu().numpy().astype(np.uint8))
    if (mask == 0).all():
        filter_num += 1
        continue
    img.save(os.path.join(output_dataset, f), quality=95)
print("Elapsed {}s, filtered {} images".format(time() - start, filter_num))
