#!/usr/bin/env python
"""
    File Name   :   s3g-vis_results
    date        :   3/12/2019
    Author      :   wenbo
    Email       :   huwenbodut@gmail.com
    Description :
                              _     _
                             ( |---/ )
                              ) . . (
________________________,--._(___Y___)_,--._______________________
                        `--'           `--'
    Modified by : Zhengyuan Su
"""
from typing import List
import plyfile
from os.path import join, isdir
import os

import numpy as np

from tqdm import tqdm
import multiprocessing as mp
from utils.config import cfg
from time import time

origin_path = 'dataset/ScanNet/val'
vis_prefix = 'vis/' + cfg.training_name
assert isdir(origin_path)
os.makedirs(vis_prefix, exist_ok=True)

# color palette for nyu40 labels
def create_color_palette():
    return np.array([
        (0, 0, 0),
        (174, 199, 232),  # wall
        (152, 223, 138),  # floor
        (31, 119, 180),  # cabinet
        (255, 187, 120),  # bed
        (188, 189, 34),  # chair
        (140, 86, 75),  # sofa
        (255, 152, 150),  # table
        (214, 39, 40),  # door
        (197, 176, 213),  # window
        (148, 103, 189),  # bookshelf
        (196, 156, 148),  # picture
        (23, 190, 207),  # counter
        (178, 76, 76),
        (247, 182, 210),  # desk
        (66, 188, 102),
        (219, 219, 141),  # curtain
        (140, 57, 197),
        (202, 185, 52),
        (51, 176, 203),
        (200, 54, 131),
        (92, 193, 61),
        (78, 71, 183),
        (172, 114, 82),
        (255, 127, 14),  # refrigerator
        (91, 163, 138),
        (153, 98, 156),
        (140, 153, 101),
        (158, 218, 229),  # shower curtain
        (100, 125, 154),
        (178, 127, 135),
        (120, 185, 128),
        (146, 111, 194),
        (44, 160, 44),  # toilet
        (112, 128, 144),  # sink
        (96, 207, 209),
        (227, 119, 194),  # bathtub
        (213, 92, 176),
        (94, 106, 211),
        (82, 84, 163),  # otherfurn
        (100, 85, 144)
    ])

reverseMapper = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39, 40])
color_palette = create_color_palette()

def visualize_one_scan(scene_name, label_pred, verbose=True):
    name = scene_name.split('/')[-1][:12]
    ori = plyfile.PlyData().read(join(origin_path, scene_name + '_vh_clean_2.ply'))

    label_pred = reverseMapper[label_pred]
    ori.elements[0].data['red'] = color_palette[label_pred][:, 0]
    ori.elements[0].data['green'] = color_palette[label_pred][:, 1]
    ori.elements[0].data['blue'] = color_palette[label_pred][:, 2]
    ori.write(join(vis_prefix, name+'.ply'))
    np.savetxt(join(vis_prefix, name+'.txt'), label_pred, fmt='%d')
    if verbose: print(f"Finish {scene_name}.")
    
def write_seg_result(
    scene_names: List[str], 
    label_pred: np.ndarray, 
    scene_offsets: List[int],
    multiprocess: bool=True):
    """
    scene_names: NumScenes
    label_pred: ndarray, (N, )
    scene_offsets: NumScenes+1 start of scene[i]
    """
    print("Begin visualize pointcloud")
    if not multiprocess:
        for id, scene in enumerate(tqdm(scene_names)):
            pred = label_pred[scene_offsets[id]:scene_offsets[id+1]]
            visualize_one_scan(scene, pred, verbose=False)
    else:
        print(f"Use multi-process. Found {mp.cpu_count()} CPU cores, will use half of them ({mp.cpu_count()//2})")
        input_list = []
        print("Preparing pipeline.")
        for id, scene in enumerate(tqdm(scene_names)):
            pred = label_pred[scene_offsets[id]:scene_offsets[id+1]]
            input_list.append((scene, pred))
        print("Launching pipeline.")
        start = time()
        p = mp.Pool(processes=mp.cpu_count() // 2)
        p.starmap(visualize_one_scan, input_list)
        p.close()
        p.join()
        print(f"Visualization finished. Elapsed {time() - start} seconds.")
        
def np_array_to_hex(array):
    array = np.asarray(array, dtype='uint32')
    array = (1 << 24) + ((array[..., 0]<<16) + (array[..., 1]<<8) + array[..., 2])
    return [hex(x)[-6:].upper() for x in array.ravel()]

if __name__ == '__main__':
    # generate color pallete
    print(np_array_to_hex(color_palette))
        