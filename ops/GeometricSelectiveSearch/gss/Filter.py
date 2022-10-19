import torch
import numpy as np
from scipy.stats import mode
import os.path as osp
from glob import glob
import multiprocessing as mp

PointThresh = 100
PseudoThresh = 1

NUM_CLASSES = 20

agg_file = glob(osp.join('aggregated_data', '*_info.pth'))

def filter_one_scene(f_agg):
    scene_name = f_agg.split('/')[-1][:12]
    
    coords, boxes, labels, pseudo_label = torch.load(f_agg)
    
    NumLabel = [np.sum(labels == i) for i in range(NUM_CLASSES)]
    NumPseudo = [np.sum(pseudo_label == i) for i in range(NUM_CLASSES)]
    CorrectPseudo = [np.sum((pseudo_label == i) * (pseudo_label == labels)) for i in range(NUM_CLASSES)]
    CorrectBoxes = np.zeros(NUM_CLASSES)
    NumFilteredBoxes = np.zeros(NUM_CLASSES)
    TotalBoxes = boxes.shape[0]
    NumBoxPseudos = []
    
    for box in boxes:
        mask = (np.prod(coords >= box[None, 0:3], -1) * np.prod(coords <= box[None, 3:6], -1)).astype('bool')
        cropped_pseudo = pseudo_label[mask]
        cropped_label = labels[mask]
        # print(cropped_label.shape)
        NumBoxPseudo = np.sum(cropped_pseudo != -100)
        NumBoxPseudos.append(NumBoxPseudo)
        if NumBoxPseudo < PseudoThresh:
            continue
        if np.all(cropped_label == -100) or np.all(cropped_pseudo == -100):
            continue
        try:
            box_label = int(mode(cropped_pseudo[cropped_pseudo != -100], axis=None)[0])
            real_label = int(mode(cropped_label[cropped_label != -100], axis=None)[0])
            NumFilteredBoxes[real_label] += 1
            if box_label == real_label: CorrectBoxes[real_label] += 1
        except:
            print(mode(cropped_label[cropped_label != -100], axis=None))
    
    NumBoxPseudos = sum(NumBoxPseudos) / len(NumBoxPseudos)
    
    print(scene_name)
    return NumLabel, NumPseudo, CorrectPseudo, CorrectBoxes, NumFilteredBoxes, TotalBoxes, NumBoxPseudos
    
from time import time
start = time()
p = mp.Pool(processes=mp.cpu_count() // 2)
stats = p.map(filter_one_scene, agg_file)
p.close()
p.join()
print(f"Filter finished. Elapsed {time() - start} seconds.")

NumLabel = np.array(list(map(lambda x: x[0], stats))) # S, 20
NumPseudo = np.array(list(map(lambda x: x[1], stats))) # S, 20
CorrectPseudo = np.array(list(map(lambda x: x[2], stats))) # S, 20
CorrectBoxes = np.array(list(map(lambda x: x[3], stats))) # S, 20
NumFilteredBoxes = np.array(list(map(lambda x: x[4], stats))) # S, 20
TotalBoxes = np.array(list(map(lambda x: x[5], stats))) # S,
NumBoxPseudos = np.array(list(map(lambda x: x[6], stats))) # S,

NumLabel = np.sum(NumLabel, 0)
NumPseudo = np.sum(NumPseudo, 0)
CorrectPseudo = np.sum(CorrectPseudo, 0)
CorrectBoxes = np.sum(CorrectBoxes, 0)
NumFilteredBoxes = np.sum(NumFilteredBoxes, 0)
TotalBoxes = np.sum(TotalBoxes, 0)
NumBoxPseudos = np.sum(NumBoxPseudos, 0)

PseudoLabelProp = NumPseudo / NumLabel
PseudoLabelQuality = CorrectPseudo / NumPseudo
BoxLabelQuality = CorrectBoxes / NumFilteredBoxes
BoxFilterProp = np.sum(NumFilteredBoxes) / TotalBoxes

def name_of_object(arg):
    # check __name__ attribute (functions)
    try:
        return arg.__name__
    except AttributeError:
        pass

    for name, value in globals().items():
        if value is arg and not name.startswith('_'):
            return name

def show(object: np.ndarray):
    print(name_of_object(object), object)

print("===============STATISTICS===============")
print("Raw value")
show(NumLabel)
show(NumPseudo)
show(CorrectPseudo)
show(CorrectBoxes)
show(NumFilteredBoxes)
show(TotalBoxes)
show(NumBoxPseudos)
print("Stats")
show(PseudoLabelProp)
show(PseudoLabelQuality)
show(BoxLabelQuality)
show(BoxFilterProp)
print("========================================")

# 598 1318 0.4537177541729894


"""

pseudo label proportion = NumPseudo / NumLabel
pseudo label quality = CorrectPseudo / NumLabel
box label quality = CorrectBox / NumBox
box pseudo average = NumBoxPseudo
class level

cropped region visualization
"""