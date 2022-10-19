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
    shapeLabels: torch.Tensor,
    coords: torch.Tensor,
    labels: torch.Tensor,
    Ylabels: torch.Tensor,
    Smask: torch.Tensor=None,
    augSmask: torch.Tensor=None):
    """
    Ulogits (B, N, C): U_seg
    Slogits (B, N, C): S_seg
    augSlogits (B, N, C): S_seg_aug
    Sdetlogits (B, N, C): S_det
    coords (B, N, 3): coords 
    boxes (B, R, 6): boxes 
    shapeLabels (B, N,): shapeLabels 
    labels (B, C): scene-level labels
    Ylabels (B, N, C): Y_seg (one_hot or all zero)
    Smask, augSmask: overlapping meta
    """
    Slogits = Slogits[:, :-1]
    return \
        WyPR_Seg_MILLoss(Ulogits, labels) + \
        WyPR_Seg_SelfTrainingLoss(Slogits, Ylabels) + \
        WyPR_Seg_CrossTransformConsistencyLoss(Slogits, augSlogits, Smask=Smask, augSmask=augSmask) + \
        WyPR_Seg_CrossTaskConsistencyLoss(coords, Slogits, Sdetlogits, boxes) + \
        WyPR_Seg_SmoothLoss(Slogits, shapeLabels)
        
@LOSS_REGISTRY.register()
def WyPR_DetLoss(
    Ulogits: torch.Tensor, 
    Slogits: torch.Tensor,
    augSlogits: torch.Tensor, 
    labels: torch.Tensor,
    Ylabels: torch.Tensor,
    Smask: torch.Tensor=None,
    augSmask: torch.Tensor=None):
    """
    Ulogits (B, R, C): U_det
    Slogits (B, R, C): S_det
    augSlogits (B, R, C): S_det_aug
    labels (B, C): scene-level labels
    Ylabels (B, R, C): Y_det (one_hot or all zero)
    Smask, augSmask: overlapping meta
    """
    return \
        WyPR_Det_MILLoss(Ulogits, labels) + \
        WyPR_Det_SelfTrainingLoss(Slogits, Ylabels) + \
        WyPR_Det_CrossTransformConsistencyLoss(Slogits, augSlogits, Smask=Smask, augSmask=augSmask)

@LOSS_REGISTRY.register()
def WyPR_Seg_MILLoss(Ulogits: torch.Tensor, labels: torch.Tensor):
    """
    Multi-instance label loss for segmentation module of WyPR.
    
    Ulogits (B, N, C): U_seg
    labels (B, C): scene-level labels
    
    both in float
    """
    scene_level_logits = Ulogits.mean(dim=-2) # B, C
    criterion = nn.BCEWithLogitsLoss()
    return criterion(scene_level_logits, labels)

@LOSS_REGISTRY.register()
def WyPR_Seg_SelfTrainingLoss(Slogits: torch.Tensor, Ylabels: torch.Tensor):
    """
    Self-training loss for segmentation module of WyPR.
    
    Slogits (B, N, C): S_seg
    Ylabels (B, N, C): Y_seg (one_hot or all zero)
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
    
    Slogits (B, N, C): S_seg
    augSlogits (B, N, C): augmented S_seg
    Smask and augSmask (B, N): need for transformation like crop and pad, mask out the overlapping part
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
    
    coords (B, N, 3): coordinates
    Sseglogits (B, N, C): S_seg
    Sdetlogits (B, R, C): S_det
    boxes (B, R, 6): gss's box
    """
    boxProb = torch.softmax(Sdetlogits, -1) # R, C
    SsegProb = torch.softmax(Sseglogits, -1)
    Losses = []
    for batch_boxes, batch_det_logits, batch_seg_logits, batch_coords in zip(boxes, boxProb, SsegProb, coords):
        for box, Blogits in zip(batch_boxes, batch_det_logits):
            mask = (torch.prod(batch_coords >= box[0:3], -1) * torch.prod(batch_coords >= box[3:6], -1)).bool()
            Losses.append(alignLogitsLoss(Blogits, batch_seg_logits.masked_select(mask)))
    loss = torch.stack(Losses).mean()
    return loss
    
@LOSS_REGISTRY.register()
def WyPR_Seg_SmoothLoss(
    Slogits: torch.Tensor, 
    shapeLabels: torch.Tensor):
    """
    Smoothness Regularization loss for segmentation module of WyPR.
    
    Slogits (B, N, C): S_seg
    shapeLabels (B, N): G
    """
    Losses = []
    for batch_logits, batch_shape in zip(Slogits, shapeLabels):
        shapeId = torch.unique(batch_shape)
        for id in shapeId:
            mask = batch_shape == id
            selectLogits = torch.softmax(batch_logits.masked_select(mask), -1)
            Losses.append(alignLogitsLoss(selectLogits, selectLogits.mean(dim=0)))
    loss = torch.stack(Losses).mean()
    return loss

# Detection Losses

@LOSS_REGISTRY.register()
def WyPR_Det_MILLoss(Ulogits: torch.Tensor, labels: torch.Tensor):
    """
    Multi-instance label loss for detection module of WyPR.
    
    Ulogits (B, R, C): U_det
    labels (B, C): scene-level labels
    
    both in float
    """
    scene_level_logits = torch.mean(Ulogits, -2)
    criterion = nn.BCEWithLogitsLoss()
    return criterion(scene_level_logits, labels)

@LOSS_REGISTRY.register()
def WyPR_Det_SelfTrainingLoss(Slogits: torch.Tensor, Ylabels: torch.Tensor):
    """
    Self-training loss for detection module of WyPR.
    
    Slogits (B, N, C): S_det
    Ylabels (B, N, C): Y_det (one_hot or all zero)
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
    
    Slogits (B, R, C): S_det
    augSlogits (B, R, C): augmented S_det
    Smask and augSmask (B, N, ): need for transformation like crop and pad, mask out the overlapping part
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