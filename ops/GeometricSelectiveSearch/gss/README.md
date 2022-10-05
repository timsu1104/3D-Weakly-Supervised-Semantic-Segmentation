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

# Acknowledgmnent

This implementation is built upon [selective_search_py](https://github.com/belltailjp/selective_search_py), which is originally released under the MIT license.

Selective Search is proposed in <a name="selective_search_ijcv"> [J. R. R. Uijlings et al., Selective Search for Object Recognition, IJCV 2013](https://ivi.fnwi.uva.nl/isis/publications/bibtexbrowser.php?key=UijlingsIJCV2013&bib=all.bib). Please check the original paper if you are interested.