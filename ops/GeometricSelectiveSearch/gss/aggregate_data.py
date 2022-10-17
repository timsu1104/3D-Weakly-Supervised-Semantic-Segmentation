import torch
import numpy as np
from scipy.stats import mode
import os.path as osp
from glob import glob
import multiprocessing as mp
import plyfile

NUM_CLASSES = 20

PointThresh = 100
PseudoThresh = 10

type = 'fv'
box_path = osp.join('computed_proposal_scannet', type)
scan_path = '/share/datasets/ScanNetv2/scans'
pseudo_path = '/home/zhengyuan/code/3D_weakly_segmentation_backbone/3DUNetWithText/dataset/ScanNet/pseudo_label/pseudo_generator_thresh0.71'
box_file = glob(osp.join(box_path, '*_prop.npy'))
pseudo_file = glob(osp.join(pseudo_path, '*_pseudo_label.pth'))
out_path = 'aggregated_data'


# Map relevant classes to {0,1,...,19}, and ignored classes to -100
remapper=np.ones(150)*(-100)
for i,x in enumerate([1,2,3,4,5,6,7,8,9,10,11,12,14,16,24,28,33,34,36,39]):
    remapper[x]=i

def filter_one_scene(f_pseudo):
    scene_name = f_pseudo.split('/')[-1][:12]
    fn_box = osp.join(box_path, scene_name + '_prop.npy')
    fn_alignmat = osp.join(scan_path, scene_name, scene_name + '.txt')
    fn_pc = osp.join(scan_path, scene_name, scene_name + '_vh_clean_2.ply')
    fn_label = osp.join(scan_path, scene_name, scene_name + '_vh_clean_2.labels.ply')
    fn_agg = osp.join(out_path, scene_name + '_info.pth')
    
    # get aligned pc
    lines = open(fn_alignmat).readlines()
    for line in lines:
        if 'axisAlignment' in line:
            axis_align_matrix = [float(x) \
                for x in line.rstrip().strip('axisAlignment = ').split(' ')]
            break
    axis_align_matrix = np.ascontiguousarray(axis_align_matrix).reshape((4,4))
    plydata = plyfile.PlyData.read(fn_pc)
    num_verts = plydata['vertex'].count
    mesh_vertices = np.zeros(shape=[num_verts, 3], dtype=np.float32)
    mesh_vertices[:,0] = plydata['vertex'].data['x']
    mesh_vertices[:,1] = plydata['vertex'].data['y']
    mesh_vertices[:,2] = plydata['vertex'].data['z']
    pts = np.ones((mesh_vertices.shape[0], 4))
    pts[:,0:3] = mesh_vertices
    pts = np.dot(pts, axis_align_matrix.transpose()) # Nx4
    coords = pts[:, :3]
    
    boxes = np.load(fn_box)
    
    labels_file=plyfile.PlyData().read(fn_label)
    labels = remapper[np.array(labels_file.elements[0]['label'])]
    pseudo_label = torch.load(f_pseudo)
    
    torch.save((coords, boxes, labels, pseudo_label), fn_agg)
    
    print(scene_name)
    
from time import time
start = time()
p = mp.Pool(processes=mp.cpu_count())
stats = p.map(filter_one_scene, pseudo_file)
p.close()
p.join()
print(f"Aggregation finished. Elapsed {time() - start} seconds.")