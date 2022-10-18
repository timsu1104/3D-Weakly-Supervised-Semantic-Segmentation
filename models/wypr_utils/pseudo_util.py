import os
import torch

def get_borders(proposal: torch.Teonsor):
    x, y, z, w, h, l = proposal
    x, y, z, w, h, l = float(x), float(y), float(z), float(w), float(h), float(l)
    x_min = x - w * 0.5
    x_max = x + w * 0.5
    y_min = y - h * 0.5
    y_max = y + h * 0.5
    z_min = z - l * 0.5
    z_max = z + l * 0.5
    return torch.tensor([x_min, x_max, y_min, y_max, z_min, z_max])

def get_vol(proposal: torch.Tensor):
    _, _, _, w, h, l = proposal
    w, h, l = float(w), float(h), float(l)
    return w * h * l

def get_iou(proposal1, proposal2):
    # w1+w2, h1+h2, l1+l2
    len_sum = torch.vstack([proposal1, proposal2])[:,3:].sum(dim=0)
    borders = torch.vstack([get_borders(proposal1), get_borders(proposal2)])
    # dimension x,y,z
    mins, _ = torch.min(borders[:,[0,2,4]], dim=0)
    maxs, _ = torch.max(borders[:,[1,3,5]], dim=0)[0]
    inter_vol = max([0.0, float(torch.prod(len_sum - (maxs - mins), dim=-1))])
    vol1 = get_vol(proposal1)
    vol2 = get_vol(proposal2)
    iou = inter_vol / (vol1 + vol2 - inter_vol)
    return iou

def get_iou_flag(p: torch.Tensor, p_Rc_star: torch.Tensor, tao: float) -> bool:
    for pr in p_Rc_star:
        iou = get_iou(p, pr)
        if iou >= tao:
            return False
    return True

def get_pseudo_labels_det(scene_label: torch.Tensor, U_det: torch.Tensor, proposals: torch.Tensor, tao: float, p2: float):
    """
    generate pseudo proposal labels for detection
    scene_label: torch.Tensor - (num_class, )
    U_det: torch.Tensor - (num_proposal, num_class)
    proposals: torch.Tensor - (num_proposal, 6)
    """
    num_above_p2 = int(U_det.size(0) * p2)
    pseudo_labels = torch.zeros(proposals.size(0), scene_label.size(0))

    for idx, i in enumerate(scene_label):
        if i == 0:
            continue
        else:
            U_det_sort = torch.sort(U_det, dim=0, descending=True)
            # logits with top p2 percent confidence with respect to class c
            Rc = U_det_sort[0][:,idx][:num_above_p2]
            # according indices
            Rc_idxs = U_det_sort[1][:,idx][:num_above_p2]
            Rc_star = [Rc[0]]
            Rc_star_idxs = [Rc_idxs[0]]
            pseudo_labels[Rc_idxs[0], idx] = 1
            for j in range(1, Rc.size(0)):
                p = proposals[Rc_idxs[j]]
                p_Rc_star = proposals[Rc_star_idxs]
                if get_iou_flag(p, p_Rc_star, tao):
                    Rc_star.append(Rc[j])
                    Rc_star_idxs.append(Rc_idxs[j])
                    pseudo_labels[Rc_idxs[j], idx] = 1

    return pseudo_labels

def get_pseudo_labels_seg(scene_label: torch.Tensor, U_seg: torch.Tensor, p1: float):
    """
    scene_label - num_class,
    U_seg - num_points, num_class
    c - num_points, num_class
    """
    num_below_p1 = int(U_seg.size(0) * p1)
    pseudo_labels = torch.zeros(U_seg.size(0), scene_label.size(0))

    c = torch.argmax(torch.mul(U_seg, scene_label.reshape(U_seg.size(0),-1))).reshape(-1, 1)
    pseudo_labels.scatter(-1, c, 1)
    #torch.gather(pseudo_labels, -1, c) = 1

    for idx, i in enumerate(scene_label):
        if i == 0:
            continue
        else:
            U_seg_sort = torch.sort(U_seg, dim=0, descending=True)
            # logits with bottom p1 percent confidence with respect to class c
            Pc = U_seg_sort[0][:,idx][:num_below_p1]
            # according indices
            Pc_idxs = U_seg_sort[1][:,idx][:num_below_p1]
            pseudo_labels[Pc_idxs, idx] = 0

    return pseudo_labels

def store_pseudo_labels_det(pseudo_labels, scene_name, path, suffix='_det_pseudo_label.txt'):
    print(scene_name)
    torch.save(pseudo_labels, os.path.join(path, scene_name + suffix))
    return