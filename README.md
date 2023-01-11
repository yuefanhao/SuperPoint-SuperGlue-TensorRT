# SuperPoint SuperGlue Acceleration
Accelerate SuperPoint and SuperGlue with TensorRT.

## Demo
<img src="image/match_image.jpg" width = "1280" height = "360"  alt="match_image" border="10" />

## Environment
* CUDA
* TensorRT
* OpenCV
* EIGEN
* yaml-cpp
## Build and Run
```bash
git clone https://github.com/yuefanhao/SuperPointSuperGlueAcceleration.git
cd SuperPointSuperGlueAcceleration
mkdir build
cd build
cmake ..
make
./superpointglueacceleration  ../config/config.yaml ../weights/ ${PWD}/../image/image0.png ${PWD}/../image/image1.png
```
