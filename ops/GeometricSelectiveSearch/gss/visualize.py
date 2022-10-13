from vis_box import write_bbox
import numpy as np
import os.path as osp, os
from glob import glob
import multiprocessing as mp

type = 'fv_inst100_p100_d300'
box_path = osp.join('computed_proposal_scannet', type)
box_file = glob(osp.join(box_path, '*_prop.npy'))
vis_path = osp.join('visualization', type)

def visualize_one_box(f):
    scene_name = f.split('/')[-1][:-9]
    out_folder = osp.join(vis_path, scene_name)
    os.makedirs(out_folder, exist_ok=True)
    fn = os.path.join('/share/datasets/ScanNetv2/scans', scene_name, scene_name + '.txt')
    
    lines = open(fn).readlines()
    for line in lines:
        if 'axisAlignment' in line:
            axis_align_matrix = [float(x) \
                for x in line.rstrip().strip('axisAlignment = ').split(' ')]
            break
    axis_align_matrix = np.ascontiguousarray(axis_align_matrix).reshape((4,4))
    
    box = np.load(f)
    assert box.shape[1] == 7
    box[:, -1] = 0
    for id, single_box in enumerate(box):
        write_bbox(single_box, 1, osp.join(out_folder, str(id) + '_pseudo_bbox.ply'))
    print(scene_name)
    
from time import time
start = time()
p = mp.Pool(processes=mp.cpu_count() // 2)
p.map(visualize_one_box, box_file)
p.close()
p.join()
print(f"Visualization finished. Elapsed {time() - start} seconds.")