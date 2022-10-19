## Getting Started
The code is structured for mainly three functionality: pre-processing (shape_det), proposal generation (gss), weakly-supervised recognition (wypr).

### Shape Detection
We use the open-source CGAL library to detect shapes from points clouds. 
This pre-precessing step needs to be done before computing proposals or launch training.
```bash
# Complie our modified C++ code, this will require CGAL
# Clone the repo in recursive model so that cgal will be downloaded
# To learn more: https://cgal.geometryfactory.com/CGAL/doc/master/Shape_detection/index.html#Shape_detection_RegionGrowing
# Use Cmake 3.1 to 3.15 (e.g., module load cmake/3.13.3/gcc.7.3.0)
cd shape_det
mkdir build; cd build
cmake -DCGAL_DIR="$(realpath ../../3rd_party/cgal/)" -DCMAKE_BUILD_TYPE=Debug ../ 
make        
# Usage: ./region_growing_on_point_set_3 input(*.xyz) output(*.ply) output(*.txt)
# To test whether it's built correctly
./region_growing_on_point_set_3 ../data/point_set_3.xyz point_set_3.ply point_set_3.txt
# You can visualize ../data/point_set_3.xyz and point_set_3.ply using tools like meshlab.
# The index assignment is saved as point_set_3.txt where 
# each row represents one shape and the last row is the un-assigned points.
cd ../..
```
**Known Issues**

0. Make sure eigen is installed `sudo apt install libeigen3-dev`.
1. `Could NOT find GMP (missing: GMP_LIBRARIES GMP_INCLUDE_DIR)`, solve with `sudo apt-get install libgmp10 libgmp-dev`.
2. `Could NOT find MPFR (missing: MPFR_LIBRARIES MPFR_INCLUDE_DIR)`, solve with `sudo apt install libcgal-dev`. ([source](https://github.com/PyMesh/PyMesh/issues/96))
3. `fatal error: GL/gl.h: No such file or directory`, try `sudo apt install mesa-common-dev`

To pre-process the ScanNet dataset, do
```bash
# 1st: Convert data from *.ply into *.xyz which CGAL can use
#      You should open some *.xyz files in meshlab to make sure things are correct
# 2nd: Generate running scripts
# Note: you need to change the `data_path` to be the absolute path of output
python shape_det/generate_scripts.py

# Running
cd shape_det/build
# Use the generated *.sh files here to detect shapes
sh *.sh
# Results will be saved in *.txt files under shape_det/build/

# Pre-compute the adjancency matrix between detected shapes
python shape_det/preprocess.py
```

For S3DIS dataset, please check the [README.md](../wypr/dataset/s3dis/README.md).

The pre-computed detected shapes can be downloaded from:
| Dataset |   url    | 
|---------|----------|
| ScanNet | [link](https://www.dropbox.com/s/a8vrkya9wtayz3h/scannet_cgal_results.zip?dl=0) |
|  S3DIS  | [link](https://www.dropbox.com/s/ctrji5bfcao65z8/s3dis_cgal_results.zip?dl=0) |

### Geometric Selective Search (GSS)
We provide standalone code to compute 3D box proposals in an unsupervised manner, together with the evaluation, and visualizaztion code.

To generate proposals for ScanNet, do
```bash
cd gss
# Compute proposals for a single policy
# Change the setting in main funtion to different policies
# eg., scannet, default policy: size
# set policy name and mask in line 150-151
python selective_search_3d_run.py \
   --split trainval \
   --dataset scannet \
   --data_path ${DATA_PATH} \
   --cgal_path ${CGAL_PATH} \
   --seg_path ${SEG_PATH} \
   --n_proc ${NUM_PROC}

python selective_search_3d_run.py --split train --dataset scannet --cgal_path ../shape_det/cgal_output --data_path /home/zhengyuan/code/3D_weakly_segmentation_backbone/3DUNetWithText/dataset/ScanNet

# [Optional] Ensemble multiple runs
python selective_search_3d_ensemble.py

# Evaluate the MABO and AR
python selective_search_3d_eval.py \
   --dataset scannet \
   --policy ${NAME_OF_POLICY} \
   --split val

python selective_search_3d_eval.py --dataset scannet --policy fv_inst100_p100_d300 --split val > test.log

```

The pre-computed 3D proposals using GSS can be found at:

| Dataset | Methods | MABO | AR | url | 
|---------|---------|------|----|-----|
| ScanNet | GSS (unsupervised) | 0.378 | 86.2 | [link](https://www.dropbox.com/s/ahgbg5zehdlpb88/scannet_gss_unsup.zip?dl=0) |
| ScanNet | GSS                | 0.409 | 89.3 | [link](https://www.dropbox.com/s/lzwk71hsmtljr0j/scannet_gss.zip?dl=0) |

Experiments result
**S, F, V** stand for size, fill, and volume respectively

| Policies | MABO | AR | AP | 
|---------|------|----|-----|
| S | 0.390 | 87.3 | 0.2407 |
| F | 0.404 | 83.7 | 0.2313 |
| V | 0.384 | 87.6 | 0.2591 |
| SV | 0.385 | 87.3 | 0.1982 |
| FV | 0.431 | 89.3 | 0.1927 |
| SF | 0.426 | 89.0 | 0.1934 |
| SFV | 0.422 | 88.5 | 0.1874 |
| SV+FV | 0.412 | 90.7 | 0.1214 |
| SV+FV (inst 100) | 0.411 | 90.2 | 0.1246 |
| FV (inst 100) | 0.433 | 88.7 | 0.1917 |
| FV (inst 100, p100, d300) | 0.427 | 88.3 | 0.3494 |

| Instance | Points number | Planar density | Volumetric density
|---------|------|----|-----|
| **Average** | **3392** | **123.90 ~ 2327.04** | **827.80 ~ 79380.88** |
| cabinet | 1781 | 63.19 ~ 3057.35 | 219.17 ~ 108824.87 |
| bed | 9921 | 115.22 ~ 1653.33 | 479.31 ~ 12965.79 |
| chair | 486 | 31.25 ~ 2407.66 | 320.68 ~ 100676.9 |
| sofa | 5901 | 169.77 ~ 1901.31 | 621.52 ~ 24796.37 |
| table | 2179 | 53.8 ~ 2364.98 | 222.18 ~ 175029.91 |
| door | 1387 | 24.88 ~ 2709.73 | 171.77 ~ 156147.32 |
| window | 3989 | 13.72 ~ 3582.11 | 69.54 ~ 204632.56 |
| bookshelf | 6451 | 185.45 ~ 3025.45 | 439.46 ~ 42573.83 |
| picture | 632 | 54.22 ~ 1976.68 | 436.73 ~ 144536.16 |
| counter | 2625 | 71.13 ~ 1731.96 | 679.29 ~ 51143.51 |
| desk | 3183 | 65.68 ~ 1593.26 | 323.09 ~ 32089.31 |
| curtain | 7182 | 85.83 ~ 2833.15 | 312.89 ~ 60672.18 |
| refrigerator | 3729 | 118.27 ~ 1898.64 | 522.83 ~ 39232.36 |
| shower curtain | 3519 | 529.03 ~ 2908.92 | 5370.79 ~ 55385.42 |
| toilet | 2062 | 139.93 ~ 2336.43 | 1142.47 ~ 24249.69 |
| sink | 752 | 132.97 ~ 1862.28 | 1228.28 ~ 41666.51 |
| bathtub | 4204 | 337.15 ~ 1176.26 | 2175.01 ~ 10614.25 |
| garbagebin | 1074 | 38.78 ~ 2867.18 | 165.42 ~ 143618.81 |


### WyPR Running
We use [Hydra]() to for configuration. For a single-run of training (e.g., Scannet):
```bash
python tools/train.py model=seg_det_net_ts \
    distrib_backend=ddp backbone=pointnet2 num_point=40000 \
    batch_size=32 learning_rate=0.003 seg_pseudo_label_th=0.9 \
    hydra.run.dir=/path/to/outputs/
```

For sweeping parameters (e.g., batch_size)
```bash
python tools/train.py model=seg_det_net_ts \
    distrib_backend=ddp backbone=pointnet2 num_point=40000 \
    batch_size=32,24,48 learning_rate=0.003 seg_pseudo_label_th=0.9 \
    hydra.sweep.dir=/path/to/outputs/ -m
```


The pre-trained WyPR models are provided here.
Numbers are evaluated on validation set.

| Dataset | Methods | mIoU | AP@IoU=0.25 | url |
|---------|---------|------|-------------|-----|
| ScanNet | WyPR | 29.6 | 18.3 | [link](link) |
| S3DIS   | WyPR | 22.3 | 19.3 | [link](link) |
