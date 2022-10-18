import shutil
from PIL import Image
import os, glob
from tqdm import tqdm
import numpy as np
from configs import cfg

Output_path = cfg.Output_path
bgcolor = cfg.bgcolor
proportion_thresh = cfg.proportion_thresh
mean_thresh = cfg.mean_thresh
color_thresh = cfg.color_thresh

input_dataset = os.path.join(Output_path, "noisy")
output_dataset = os.path.join(Output_path, "clean")
if os.path.exists(output_dataset):
    shutil.rmtree(output_dataset)
os.mkdir(output_dataset)

def bgcolor_white(img: np.ndarray, p:float=0.6) -> bool:
    top, bottom = img[:1].reshape(-1, 3), img[-1:].reshape(-1, 3)
    left, right = img[1:-1, :1].reshape(-1, 3), img[1:-1, -1:].reshape(-1, 3)
    side = np.concatenate([top, bottom, left, right], axis=0)
    white_part = np.sum(np.prod(side >= color_thresh, axis=-1)).item()
    return white_part / side.shape[0] > p and np.mean(img) <= mean_thresh

def bgcolor_green(img: np.ndarray, p:float=0.6) -> bool:
    top, bottom = img[:1].reshape(-1, 3), img[-1:].reshape(-1, 3)
    left, right = img[1:-1, :1].reshape(-1, 3), img[1:-1, -1:].reshape(-1, 3)
    side = np.concatenate([top, bottom, left, right], axis=0)
    green_part = np.sum(np.prod(side <= color_thresh, axis=-1)).item()
    return green_part / side.shape[0] > p and np.mean(img) >= mean_thresh

def bgcontrast(img: np.ndarray) -> bool:
    return (np.max(np.mean(img,axis=-1)) - np.min(np.mean(img,axis=-1)) > 100)

filtered_num = 0
filtered_file = []
Files = sorted(glob.glob(os.path.join(input_dataset, '*.jpg')))
for rgb_file in tqdm(Files):
    img = np.array(Image.open(rgb_file))
    if bgcolor == 'white':
        if not bgcolor_white(img, proportion_thresh): # Upper and bottom line should be white
            filtered_num += 1
            filtered_file.append(rgb_file)
        else:
            img = Image.fromarray(img)
            img.save(os.path.join(output_dataset, rgb_file.split('/')[-1]))
    else:
        if not bgcolor_green(img, proportion_thresh): # Upper and bottom line should be white
            filtered_num += 1
            filtered_file.append(rgb_file)
        else:
            img = Image.fromarray(img)
            img.save(os.path.join(output_dataset, rgb_file.split('/')[-1]))

print("Filtered {} images, rest saved in {}".format(filtered_num, output_dataset))