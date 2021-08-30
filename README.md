
# PCS (Point Cloud Segmentation Library)

![Linux Build Status](https://github.com/aleksrgarkusha/pcs/actions/workflows/cmake.yml/badge.svg)

C++ library for semantic segmentation of large scale point clouds with python wrappers

The unofficial implementaion of the paper:  
*"Classification of Aerial Photogrammetric 3D Point Clouds" ISPRS 2017
    C. Becker, N. Haeni, E. Rosinskaya, E. d'Angelo, C. Strecha*

Segmentation results for **building.xyz** dataset:  
![Building point cloud segmentation](/assets/building.png "Segmented point cloud")

Segmentation results for **ankeny.xyz** dataset:  
![Ankeny point cloud segmentation](/assets/ankeny.png "Segmented point cloud")  

## Prerequisites

* A compiler with C++11 support
* CMake >= 3.4
* Eigen3 library

## Build instructions
Clone the repository and build python extension with followinng commands:  
```bash
git clone --recursive https://github.com/aleksrgarkusha/pcs.git
cd pcs
python3 setup.py build_ext --inplace
```

After building is finished you can launch `jupyter notebook` from current folder and run `point_cloud_segmentation.ipynb` script for example of libray usage

## Citation 
```
@misc{
  title={Classification of Aerial Photogrammetric 3D Point Clouds},
  author={C. Becker, N. Haeni, E. Rosinskaya, E. d'Angelo, C. Strecha},
  booktitle={Journal of Photogrammetry and Remote Sensing (ISPRS)},
  year={2017}
}
```

