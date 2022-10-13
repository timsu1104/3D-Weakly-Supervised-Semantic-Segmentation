import torch
from torch import nn
from utils.registry import MODEL_REGISTRY

from .components import cropBox, MattingModule, Voxelizer

@MODEL_REGISTRY.register()
class Projector(nn.Module):
    """
    Matting the pointcloud
    """
    def __init__(self, in_channels, out_channels=2, resolution=256) -> None:
        super().__init__()
        self.matting = MattingModule(in_channels, out_channels)
        self.voxelizer = Voxelizer(out_channels, resolution=resolution)
    
    def forward(self, coords, feats,pseudo_class, boxes, transform, view='HWZ'):
        cropped_coords, cropped_feats,batch_lens,dominate_class,box_class = cropBox(coords, feats, pseudo_class, boxes, transform)
        segmented_coords, segmented_feats,seg_box_class= self.matting(cropped_coords, cropped_feats,dominate_class,box_class)
        #print(box_class)
        #print(cropped_coords,segmented_coords)
        masks,img_class = self.voxelizer(segmented_coords, segmented_feats,seg_box_class, view=view)
        return masks,img_class

if __name__ == '__main__':
    from torch.autograd import Variable
    import torch.backends.cudnn as cudnn
    import numpy as np
    import plyfile
    cudnn.benchmark = False
    projector = Projector(3, resolution=64).cuda()
    projector.train()
    
    sample_data = torch.load('/home/zhengyuan/code/3D_weakly_segmentation_backbone/3DUNetWithText/dataset/ScanNet/val_processed/scene0015_00_vh_clean_2.pth')
    boxes = np.load('/home/zhengyuan/code/3D_weakly_segmentation_backbone/3DUNetWithText/ops/GeometricSelectiveSearch/gss/computed_proposal_scannet/fv_inst100_p100_d300/scene0015_00_prop.npy')
    pointcloud_file=plyfile.PlyData().read('/share/suzhengyuan/data/ScanNetv2/scan/scene0015_00/scene0015_00_vh_clean_2.ply')
    pointcloud=np.array([list(x) for x in pointcloud_file.elements[0]])
    
    lines = open('/share/suzhengyuan/data/ScanNetv2/scan/scene0015_00/scene0015_00.txt').readlines()
    for line in lines:
        if 'axisAlignment' in line:
            axis_align_matrix = [float(x) \
                for x in line.rstrip().strip('axisAlignment = ').split(' ')]
            break
    axis_align_matrix = np.array(axis_align_matrix).reshape((4,4))
    pts = np.ones((pointcloud.shape[0], 4))
    pts[:,0:3] = pointcloud[:,0:3]
    pts = np.dot(pts, axis_align_matrix.transpose()) # Nx4
    pointcloud[:,0:3] = pts[:,0:3]
    
    coords, color, _ = sample_data
    coords = torch.from_numpy(pointcloud[:,:3]).cuda()
    color = Variable(torch.from_numpy(color).cuda(), requires_grad=True)
    boxes = torch.from_numpy(boxes).cuda()
    axis_align_matrix = torch.from_numpy(axis_align_matrix).cuda().unsqueeze(0).float()

    batch_inds = torch.zeros((coords.size(0), 1)).cuda()
    coords = torch.cat([coords, batch_inds], dim=-1)
    batch_inds = torch.zeros((boxes.size(0), 1)).cuda()
    boxes = torch.cat([boxes, batch_inds], dim=-1)
    
    print('Coords', coords.size())
    print('Feats', color.size())
    print('Boxes', boxes.size())
    
    torch.autograd.set_detect_anomaly(True)
    from time import time
    start = time()
    mask = projector(coords, color, boxes, [axis_align_matrix, torch.zeros((1, 3)).cuda(), torch.eye(3).cuda().unsqueeze(0), torch.zeros((1, 3)).cuda()], view='HWZ')
    end = time()
    print('Mask', mask.size())
    print('Elapsed', end - start, 'seconds.')
    
    start = time()
    mask.backward(
        torch.FloatTensor(*mask.size()).fill_(1).cuda()
    )
    end = time()
    print(color.grad)
    print('Elapsed', end - start, 'seconds.')
    