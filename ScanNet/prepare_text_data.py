# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import glob, multiprocessing as mp
import os, json
import os.path as osp

files = sorted(glob.glob('train/*_vh_clean_2.ply')) + sorted(glob.glob('val/*_vh_clean_2.ply'))

text_description = {
    'train': json.load(open('ScanRefer_filtered_train.json', 'r')), 
    'val': json.load(open('ScanRefer_filtered_val.json', 'r'))
}

def f(fn: str):
    split = fn.split('/')[0]
    file_name = fn[len(split) + 1:]
    scene_name = file_name[:-15]
    selected_text = list(
        map(lambda desc: desc['description'], 
        filter(lambda desc: desc['scene_id'] == scene_name, 
        text_description[split]
        )))

    if not osp.exists(split + '_processed'):
        os.makedirs(split + '_processed')
    with open(osp.join(split + '_processed', scene_name + '_text.json'), 'w') as f:
        json.dump(selected_text, f)

    print(fn)

p = mp.Pool(processes=mp.cpu_count())
p.map(f,files)
p.close()
p.join()
