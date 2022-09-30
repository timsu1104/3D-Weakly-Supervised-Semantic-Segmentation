import shutil
from PIL import Image
import os, glob
from tqdm import tqdm
import numpy as np
from configs import cfg

Output_path = cfg.Output_path

input_dataset = os.path.join(Output_path, "noisy")
output_dataset = os.path.join(Output_path, "clean")
if os.path.exists(output_dataset):
    shutil.rmtree(output_dataset)
os.mkdir(output_dataset)

def bgcolor(img: np.ndarray, p:float=0.6) -> bool:
    top, bottom = img[:5].reshape(-1, 3), img[-5:].reshape(-1, 3)
    left, right = img[5:-5, :5].reshape(-1, 3), img[5:-5, -5:].reshape(-1, 3)
    side = np.concatenate([top, bottom, left, right], axis=0)
    white_part = np.sum(np.prod(side >= 240, axis=-1)).item()
    return white_part / side.shape[0] > p

filtered_num = 0
filtered_file = []
Files = sorted(glob.glob(os.path.join(input_dataset, '*.jpg')))
for rgb_file in tqdm(Files):
    img = np.array(Image.open(rgb_file))
    if not bgcolor(img): # Upper and bottom line should be white
        filtered_num += 1
        filtered_file.append(rgb_file)
    else:
        img = Image.fromarray(img)
        img.save(os.path.join(output_dataset, rgb_file.split('/')[-1]))

print("Filtered {} images, rest saved in {}".format(filtered_num, output_dataset))