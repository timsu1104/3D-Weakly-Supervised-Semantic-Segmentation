from vis_box import write_bbox
import numpy as np
import os.path as osp, os
from glob import glob
import multiprocessing as mp
import plyfile

type = 'fv'
box_path = osp.join('computed_proposal_scannet', type)
box_file = glob(osp.join(box_path, '*_prop.npy'))
vis_path = osp.join('visualization', type)

def visualize_one_scene(f):
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
    
    plydata = plyfile.PlyData.read(osp.join('/share/datasets/ScanNetv2/scans', scene_name, scene_name + '_vh_clean_2.ply'))
    num_verts = plydata['vertex'].count
    mesh_vertices = np.zeros(shape=[num_verts, 3], dtype=np.float32)
    mesh_vertices[:,0] = plydata['vertex'].data['x']
    mesh_vertices[:,1] = plydata['vertex'].data['y']
    mesh_vertices[:,2] = plydata['vertex'].data['z']

    pts = np.ones((mesh_vertices.shape[0], 4))
    pts[:,0:3] = mesh_vertices
    pts = np.dot(pts, axis_align_matrix.transpose()) # Nx4
    
     
    plydata.write(osp.join(out_folder, scene_name + '_aligned.ply'))
    
    box = np.load(f)
    assert box.shape[1] == 7
    box[:, -1] = 0
    for id, single_box in enumerate(box):
        write_bbox(single_box, 1, osp.join(out_folder, str(id) + '_pseudo_bbox.ply'))
    print(scene_name)
    
from time import time
start = time()
p = mp.Pool(processes=mp.cpu_count() // 2)
p.map(visualize_one_scene, box_file)
p.close()
p.join()
print(f"Visualization finished. Elapsed {time() - start} seconds.")