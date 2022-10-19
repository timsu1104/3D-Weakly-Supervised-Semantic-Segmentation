import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.registry import LOSS_REGISTRY

@LOSS_REGISTRY.register()
def TextContrastive(pc: torch.Tensor, text: torch.Tensor, has_text):
    """
    pc: B, m
    text: num_text, m
    """

    if isinstance(text, int) or len(text) == 0:
        return 0
    
    assert text.ndim == 2, text.size()
    text_without_index = text[:,:-1]
    labels = text[:,-1].long()
    similarity = text_without_index @ pc.T # num_text, B
    # similarity[i,j] -- the similarity between text i and pc feature j
    labels = labels.unsqueeze(0);
    similarity = similarity.unsqueeze(0);
    similarity = similarity / torch.norm(similarity)

    contrast_loss = F.cross_entropy(similarity.transpose(1,2), labels)
    return contrast_loss

@LOSS_REGISTRY.register()
def Classification(logits: torch.Tensor, labels: torch.Tensor):
    """
    Loss for scene level classification and point level classification.
    scene level: logits (B, C), labels (B, C)
    scene level: logits (N, C), labels (N, )
    """
    if labels.ndim == 2:
        return F.multilabel_soft_margin_loss(logits, labels)
    elif labels.ndim == 1:
        mask = labels != -100
        logits_for_loss = logits[mask]
        labels_for_loss = labels[mask]
        return F.cross_entropy(logits_for_loss, labels_for_loss)
    

# WyPR losses

# Segmentation Losses

@LOSS_REGISTRY.register()
def WyPR_SegLoss(
    Ulogits: torch.Tensor, 
    Slogits: torch.Tensor,
    augSlogits: torch.Tensor, 
    Sdetlogits: torch.Tensor, 
    boxes: torch.Tensor,
    Planes: torch.Tensor,
    coords: torch.Tensor,
    labels: torch.Tensor,
    Ylabels: torch.Tensor,
    Smask: torch.Tensor=None,
    augSmask: torch.Tensor=None):
    """
    Ulogits (N, C+1): U_seg
    Slogits (N, C+1): S_seg
    labels (B, C): scene-level labels
    Ylabels (N, C): Y_seg (one_hot or all zero)
    """
    Slogits = Slogits[:, :-1]
    return \
        WyPR_Seg_MILLoss(Ulogits, labels) + \
        WyPR_Seg_SelfTrainingLoss(Slogits, Ylabels) + \
        WyPR_Seg_CrossTransformConsistencyLoss(Slogits, augSlogits, Smask=Smask, augSmask=augSmask) + \
        WyPR_Seg_CrossTaskConsistencyLoss(coords, Slogits, Sdetlogits, boxes) + \
        WyPR_Seg_SmoothLoss(Slogits, Planes)
        
@LOSS_REGISTRY.register()
def WyPR_DetLoss(
    Ulogits: torch.Tensor, 
    Slogits: torch.Tensor,
    augSlogits: torch.Tensor, 
    labels: torch.Tensor,
    Ylabels: torch.Tensor,
    Smask: torch.Tensor=None,
    augSmask: torch.Tensor=None):
    return \
        WyPR_Det_MILLoss(Ulogits, labels) + \
        WyPR_Det_SelfTrainingLoss(Slogits, Ylabels) + \
        WyPR_Det_CrossTransformConsistencyLoss(Slogits, augSlogits, Smask=Smask, augSmask=augSmask)

@LOSS_REGISTRY.register()
def WyPR_Seg_MILLoss(Ulogits: torch.Tensor, labels: torch.Tensor):
    """
    Multi-instance label loss for segmentation module of WyPR.
    
    Ulogits (N, C+1): U_seg
    labels (B, C): scene-level labels
    
    both in float
    """
    scene_level_logits = []
    batch_ids = torch.unique(Ulogits[:, -1])
    for batch_id in batch_ids:
        batch_mask = Ulogits[:, -1] == batch_id
        scene_level_logit = torch.mean(Ulogits[batch_mask, :-1], dim=0)
        scene_level_logits.append(scene_level_logit)
    scene_level_logits = torch.stack(scene_level_logits)
    
    criterion = nn.BCEWithLogitsLoss()
    return criterion(scene_level_logits, labels)

@LOSS_REGISTRY.register()
def WyPR_Seg_SelfTrainingLoss(Slogits: torch.Tensor, Ylabels: torch.Tensor):
    """
    Self-training loss for segmentation module of WyPR.
    
    Slogits (N, C): S_seg
    Ylabels (N, C): Y_seg (one_hot or all zero)
    """
    mask = Ylabels.sum(dim=-1) == 1 # (N, )
    pseudoLabels = Ylabels[mask]
    selectedLogits = Slogits[mask]
    
    criterion = nn.CrossEntropyLoss()
    return criterion(selectedLogits, pseudoLabels)

@LOSS_REGISTRY.register()
def WyPR_Seg_CrossTransformConsistencyLoss(
    Slogits: torch.Tensor, 
    augSlogits: torch.Tensor,
    Smask: torch.Tensor=None,
    augSmask: torch.Tensor=None):
    """
    Cross-transformation Consistency loss for segmentation module of WyPR.
    
    Slogits (N, C): S_seg
    augSlogits (N, C): augmented S_seg
    Smask and augSmask (N, ): need for transformation like crop and pad, mask out the overlapping part
    """
    if Smask is not None:
        Slogits = Slogits[Smask]
    if augSmask is not None:
        augSlogits = augSlogits[augSmask]
    assert (Slogits.size() == augSlogits.size()).all()
    
    Slogits = F.log_softmax(Slogits, dim=-1)
    augSlogits = F.softmax(augSlogits, dim=-1)
    
    criterion = nn.KLDivLoss(reduction="batchmean")
    return criterion(Slogits, augSlogits)

def alignLogitsLoss(pointLogits: torch.Tensor, globalLogits):
    return torch.mean(
        torch.sum(
            - globalLogits[None] * pointLogits.log(), -1
        )
    )

@LOSS_REGISTRY.register()
def WyPR_Seg_CrossTaskConsistencyLoss(
    coords: torch.Tensor,
    Sseglogits: torch.Tensor, 
    Sdetlogits: torch.Tensor, 
    boxes: torch.Tensor):
    """
    Cross-task Consistency loss for segmentation module of WyPR.
    
    coords (N, 3): coordinates
    Sseglogits (N, C): S_seg
    Sdetlogits (N, C): S_det
    boxes (R, 6): need for transformation like crop and pad, mask out the overlapping part
    """
    boxProb = torch.softmax(Sdetlogits, -1) # N, C
    SsegProb = torch.softmax(Sseglogits, -1)
    Losses = []
    for box, Blogits in zip(boxes, boxProb):
        mask = (torch.prod(coords >= box[0:3], -1) * torch.prod(coords >= box[3:6], -1)).bool()
        Losses.append(alignLogitsLoss(Blogits, SsegProb[mask]))
    loss = torch.stack(Losses).mean()
    return loss
    
@LOSS_REGISTRY.register()
def WyPR_Seg_SmoothLoss(
    Slogits: torch.Tensor, 
    Planes: torch.Tensor):
    """
    Smoothness Regularization loss for segmentation module of WyPR.
    
    Slogits (N, C): S_seg
    Planes (G, NPts): need for transformation like crop and pad, mask out the overlapping part
    """
    Losses = []
    for plane in Planes:
        selectLogits = torch.softmax(Slogits[plane], -1)
        Losses.append(alignLogitsLoss(selectLogits, selectLogits.mean(dim=0)))
    loss = torch.stack(Losses).mean()
    return loss

# Detection Losses

@LOSS_REGISTRY.register()
def WyPR_Det_MILLoss(Ulogits: torch.Tensor, labels: torch.Tensor):
    """
    Multi-instance label loss for detection module of WyPR.
    
    Ulogits (R, C+1): U_det
    labels (B, C): scene-level labels
    
    both in float
    """
    scene_level_logits = []
    batch_ids = torch.unique(Ulogits[:, -1])
    for batch_id in batch_ids:
        batch_mask = Ulogits[:, -1] == batch_id
        scene_level_logit = torch.mean(Ulogits[batch_mask, :-1], dim=0)
        scene_level_logits.append(scene_level_logit)
    scene_level_logits = torch.stack(scene_level_logits)
    
    criterion = nn.BCEWithLogitsLoss()
    return criterion(scene_level_logits, labels)

@LOSS_REGISTRY.register()
def WyPR_Det_SelfTrainingLoss(Slogits: torch.Tensor, Ylabels: torch.Tensor):
    """
    Self-training loss for detection module of WyPR.
    
    Slogits (N, C): S_det
    Ylabels (N, C): Y_det (one_hot or all zero)
    """
    mask = Ylabels.sum(dim=-1) == 1 # (N, )
    pseudoLabels = Ylabels[mask]
    selectedLogits = Slogits[mask]
    
    criterion = nn.CrossEntropyLoss()
    return criterion(selectedLogits, pseudoLabels)

@LOSS_REGISTRY.register()
def WyPR_Det_CrossTransformConsistencyLoss(
    Slogits: torch.Tensor, 
    augSlogits: torch.Tensor,
    Smask: torch.Tensor=None,
    augSmask: torch.Tensor=None):
    """
    Cross-transformation Consistency loss for detection module of WyPR.
    
    Slogits (R, C): S_det
    augSlogits (R, C): augmented S_det
    Smask and augSmask (N, ): need for transformation like crop and pad, mask out the overlapping part
    """
    if Smask is not None:
        Slogits = Slogits[Smask]
    if augSmask is not None:
        augSlogits = augSlogits[augSmask]
    assert (Slogits.size() == augSlogits.size()).all()
    
    Slogits = F.log_softmax(Slogits, dim=-1)
    augSlogits = F.softmax(augSlogits, dim=-1)
    
    criterion = nn.KLDivLoss(reduction="batchmean")
    return criterion(Slogits, augSlogits)