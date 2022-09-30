from PIL import Image
import numpy as np
import os, shutil
from os.path import split, splitext, exists, join
from glob import glob
from tqdm import tqdm
from configs import cfg

Output_path = cfg.Output_path

input_dataset = os.path.join(Output_path, "clean")
output_dataset = os.path.join(Output_path, "mask")

if exists(output_dataset):
    shutil.rmtree(output_dataset)
os.mkdir(output_dataset)

rgb_files = glob(join(input_dataset, '*.jpg'))
for rgb_file in tqdm(rgb_files):
    fn = splitext(split(rgb_file)[-1])[0]
    img = Image.open(rgb_file)
    img = np.array(img)
    mask = 255 * np.ones_like(img)
    bg = np.nonzero(1 - np.prod(img <= 240, axis=-1))
    mask[bg] = 0
    mask = Image.fromarray(mask)
    mask.save(join(output_dataset, fn + '.jpg'), quality=95)