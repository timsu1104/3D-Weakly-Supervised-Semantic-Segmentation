import torch
import torch.nn.functional as F
from utils.registry import LOSS_REGISTRY

@LOSS_REGISTRY.register()
def TextContrastive(pc: torch.Tensor, text: torch.Tensor, has_text):
    """
    pc: B, m
    text: B', num_text, m
    """
    if has_text.size(0) == 0:
        return 0
    assert text.ndim == 3, text.size()
    similarity = text @ pc.T # B', num_text, B
    num_text = similarity.size(1)
    labels = torch.tile(has_text[:, None], (1, num_text))
    contrast_loss = F.cross_entropy(similarity.transpose(1, 2), labels)
    return contrast_loss

@LOSS_REGISTRY.register()
def Classification(global_logits: torch.Tensor, labels: torch.Tensor):
    return F.multilabel_soft_margin_loss(global_logits, labels)