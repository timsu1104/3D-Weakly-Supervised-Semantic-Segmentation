import os
import torch
from dataset.data import train_files

def get_pseudo_labels(logits: torch.Tensor, threshold: float=0.5, show_stats=False):
    """
    get pseudo labels for logits
    """
    if show_stats:
        print("STATISTICS")
        print(f"Confidence ranges from {logits.min()} to {logits.max()}, detail as below. ")
        sort_logits = torch.sort(logits.flatten(), descending=True)[0]
        pred_num = sort_logits.size(0)
        percentages = [1, 2, 3, 5, 10, 20, 30, 50, 70]
        for per in percentages:
            print(f"{per}% {sort_logits[pred_num // 100 * per]}")
    
    conf, pseudo_labels = logits.max(dim=-1)
    pseudo_labels[conf < threshold] = -100
    num_pseudo_labels = torch.sum(conf >= threshold)
    return pseudo_labels, num_pseudo_labels

def assess_label_quality(pseudo_labels, labels):
    mask = pseudo_labels != -100
    correct = torch.sum(pseudo_labels[mask] == labels[mask])
    total = torch.sum(mask)
    return correct, total

def store_pseudo_label(pseudo_labels, scene_names, batch_offset, path, suffix='_pseudo_label.pth'):

    for b, scene_name in enumerate(scene_names):
        pseudo_label = pseudo_labels[batch_offset[b] : batch_offset[b+1]]
        torch.save(pseudo_label, os.path.join(path, scene_name + suffix))
