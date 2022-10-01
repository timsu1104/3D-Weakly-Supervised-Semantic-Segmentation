from os import listdir, makedirs
import numpy as np
import time
import torch
import pickle
from os.path import *
import multiprocessing as mp
from sklearn.neighbors import KDTree
from functools import partial

# PLY reader
import sys, os
sys.path.append(os.getcwd())

train_path = 'train_processed'
test_path = 'val_processed'

print('\nPreparing ply files')
t0 = time.time()

# List of training files
train_files = np.sort([join(train_path, f) for f in listdir(train_path) if f[-4:] == '.pth'])

# Add test files
test_files = np.sort([join(test_path, f) for f in listdir(test_path) if f[-4:] == '.pth'])

def processing_single_KDTree(file_path:str, split='train'):

    # Restart timer
    t0 = time.time()

    # get cloud name and split
    cloud_name = file_path.split('/')[-1][:-15]

    # Name of the input files
    KDTree_file = join(split + '_processed', '{:s}_KDTree.pkl'.format(cloud_name))

    # Check if inputs have already been computed
    if not isfile(KDTree_file):

        # Read ply file
        points = torch.load(file_path)[0]

        # Get chosen neighborhoods
        search_tree = KDTree(points, leaf_size=50)

        # Save KDTree
        with open(KDTree_file, 'wb') as f:
            pickle.dump(search_tree, f)


    print(file_path, "completed KDTree with time {} sec".format(time.time() - t0))

mpPool = mp.Pool(mp.cpu_count() // 2)
mpPool.map(partial(processing_single_KDTree, split='train'), train_files)
mpPool.close()
mpPool.join()
mpPool = mp.Pool(mp.cpu_count() // 2)
mpPool.map(partial(processing_single_KDTree, split='val'), test_files)
mpPool.close()
mpPool.join()

print('Done in {:.1f}s'.format(time.time() - t0))
