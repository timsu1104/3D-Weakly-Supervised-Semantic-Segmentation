import torch
from torch import nn
import sparseconvnet as scn
from dataset.data import NUM_CLASSES

def cropBox(coords: torch.Tensor, feats: torch.Tensor, pseudo_class:torch.Tensor, boxes: torch.Tensor, transform: tuple):
    """
    coords: (N, 3+1), 1 for batch
    feats: (N, C)
    box: (M, 6+1), 1 for batch
    transform:
        axis_align_matrix: (B, 4, 4)
        -center, @rot.inv, +offset
    
    Return
    --------
    cropped cloud, (N', 3+1), (N', C)
    """
    device = coords.device
    coords_pool = []
    feats_pool = []
    dominate_class=[]
    box_dominate_class=[]
    axis_align_matrix, centers, rots, offsets = transform
    for id, box in enumerate(boxes):
        center = box[:3]
        length = box[3:6]
        mincoords = center - length / 2
        maxcoords = center + length / 2
        batch_id = box[-1]
        
        batch_mask = (coords[:, -1] == batch_id)
        batch_pc = coords[batch_mask, :3]
        #print(feats.size)
        # batch_feats = feats[batch_mask]
        
        batch_class = pseudo_class[batch_mask]
        batch_pc = ((batch_pc \
                     - offsets[batch_id.long()]) \
                    @ rots[batch_id.long()] \
                    + centers[batch_id.long()])
        batch_pc = torch.cat([batch_pc, torch.ones((batch_pc.size(0), 1), device=device)], -1)
        batch_pc = batch_pc @ axis_align_matrix[batch_id.long()].T
        
        selected_mask = (torch.prod(batch_pc[:, :3] >= mincoords, -1) * torch.prod(batch_pc[:, :3] <= maxcoords, -1)).bool()
        feats_sel_mask = batch_mask.masked_scatter(batch_mask, selected_mask)
        
        cropped_feats = feats[feats_sel_mask]
        cropped_coords = batch_pc[selected_mask]
        cropped_class = batch_class[selected_mask]
        # centering
        cropped_coords[:, :3] -= cropped_coords[:, :3].min(0)[0]
        cropped_coords[:, :3] /= (cropped_coords[:, :3].max(0)[0] - cropped_coords[:, :3].min(0)[0])
        cropped_coords[:, -1] = id
        
        #Get the main class in this box, for matting
        #print(cropped_feats.shape)
        box_logits=cropped_class.mean(dim=0)
        print(box_logits)
        #print(box_logits)
        max_logits,max_cls=box_logits.max(0)
        dominate_class.append(torch.full((cropped_class.size(0),1),max_cls, device=device))
        box_dominate_class.append(max_cls)
        coords_pool.append(cropped_coords)
        #TODO：change back to feats
        feats_pool.append(cropped_class)
    batch_lens=[]
    for feat in feats_pool:
        batch_lens.append(feat.size(0))
    new_coords = torch.cat(coords_pool)
    new_feats = torch.cat(feats_pool)
    dominate_class=torch.cat(dominate_class)
    box_dominate_class=torch.LongTensor(box_dominate_class).to(device)
    return new_coords, new_feats,batch_lens,dominate_class,box_dominate_class

class MattingModule(nn.Module):
    """
    Matting the pointcloud
    """
    def __init__(self, in_channels, out_channels=2) -> None:
        super().__init__()

        #TODO:currently the out_channel is assumed as 1
        self.model = nn.Linear(in_channels, out_channels*NUM_CLASSES)
        self.out_channels=out_channels
    def forward(self, coords: torch.Tensor, feats: torch.Tensor, dominate_class:torch.Tensor,box_dominate_class):
        x=self.model(feats)
        x=torch.sigmoid(x)
        print(box_dominate_class)
        
        x=x.gather(1,dominate_class)#get the mask of the dominate class
        #print(x.shape)
        mask=(x>0.5)
        new_coords=coords[mask.squeeze(1),:]
        out_feat=x[mask]
        if(x.nelement()!=0):
            out_feat=x[mask].unsqueeze(1)
        #print(new_coords[:,-1])
        appear_mask=torch.bincount(new_coords[:,-1].long(),minlength=box_dominate_class.shape[0])
        appear_mask=(appear_mask>=1)#Yyou can SET LEAST AVALIABLE image size heare
        return new_coords, out_feat,box_dominate_class[appear_mask]
class DirectMattingModule(nn.Module):
    """
    Matting the pointcloud
    """
    def __init__(self, in_channels, out_channels=2) -> None:
        super().__init__()

        #TODO:currently the out_channel is assumed as 1
        #self.model = nn.Linear(in_channels, out_channels*NUM_CLASSES)
        self.out_channels=out_channels
    def forward(self, coords: torch.Tensor, feats: torch.Tensor, dominate_class:torch.Tensor,box_dominate_class):

        x=torch.sigmoid(feats)
        print(box_dominate_class)
        
        x=x.gather(1,dominate_class)#get the mask of the dominate class
        #print(x.shape)
        mask=(x>0.5)
        new_coords=coords[mask.squeeze(1),:]
        out_feat=x[mask]
        if(x.nelement()!=0):
            out_feat=x[mask].unsqueeze(1)
        #print(new_coords[:,-1])
        appear_mask=torch.bincount(new_coords[:,-1].long(),minlength=box_dominate_class.shape[0])
        appear_mask=(appear_mask>=1)#Yyou can SET LEAST AVALIABLE image size heare
        return new_coords, out_feat,box_dominate_class[appear_mask]
class Voxelizer(nn.Module):
    """
    Parameters
    -------
    coords, feats
    
    Return
    -------
    view_mask: (B', C, H, W)
    """
    def __init__(self, channels, resolution=256) -> None:
        super().__init__()
        self.res = resolution
        self.voxelizer = scn.Sequential(
            scn.InputLayer(3, resolution, mode=4),
            scn.SparseToDense(3, channels)
        )
    # maxpooling
    def forward(self, coords: torch.Tensor, feats: torch.Tensor,box_class, view='HWZ'):
        coords[:, :-1] = coords[:, :-1] * self.res
        #print("Start voxelize")
        voxel = self.voxelizer([coords, feats]) # [B, C, H, W, Z]
        
        _, _, H, W, Z = voxel.size()
        #print(voxel.size())
        assert H == W == Z == self.res
        assert len(view) > 0, "view not selected!"
        view_cnt=0
        view_mask = []
        if 'H' in view:
            view_mask.append(torch.max(voxel, dim=-3)[0])
            view_cnt+=1
        if 'W' in view:
            view_mask.append(torch.max(voxel, dim=-2)[0])
            view_cnt+=1
        if 'Z' in view:
            view_mask.append(torch.max(voxel, dim=-1)[0])
            view_cnt+=1
        img_class=box_class.repeat(view_cnt)
        view_mask = torch.cat(view_mask) if len(view_mask) > 0 else view_mask[0]
        return view_mask,img_class
    
if __name__ == '__main__':
    import numpy as np
    box = np.load('/home/zhengyuan/code/3D_weakly_segmentation_backbone/3DUNetWithText/ops/GeometricSelectiveSearch/gss/computed_proposal_scannet/fv_inst100_p100_d300/scene0015_00_prop.npy')
    print(box.shape)
    print(box[:5])