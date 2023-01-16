# SuperPoint SuperGlue TensorRT
Accelerate SuperPoint and SuperGlue with TensorRT.

## Demo
<img src="image/match_image.png" width = "640" height = "240"  alt="match_image" border="10" />

* image pairs are from the freiburg_sequence.

## Baseline(ToDo)

| Image Size: 320 x 240  | RTX3090 | RTX3080 | Quadro P620 | Jetson Nano | Jetson TX2 NX |  
|:----------------------:|:-------:|:-------:|:-----------:|:-----------:|:-------------:|
| SuperPoint (250 points)|         |         | 13.61 MS    |             |               |
| SuperPoint (257 points)|         |         | 13.32 MS    |             |               |
| SuperGlue (256 dims)   |         |         | 58.83 MS    |             |               |

* When first run the inference, there will cost a lot of time in engine buiding and context init. So the speed baseline will only statistic except the first.

- [ ] Compare the performance and speed with the offical [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork).

## Docker(Recommand)
```bash
docker pull yuefan2022/tensorrt-ubuntu20.04-cuda11.6:latest
docker run -it --env DISPLAY=$DISPLAY --volume /tmp/.X11-unix:/tmp/.X11-unix --privileged --runtime nvidia --gpus all --volume ${PWD}:/workspace --workdir /workspace --name tensorrt yuefan2022/tensorrt-ubuntu20.04-cuda11.6:latest /bin/bash
```

## Environment Required
* CUDA
* TensorRT
* OpenCV
* EIGEN
* yaml-cpp

## Convert Model(Optional)
```bash
python convert2onnx/convert_superpoint_to_onnx.py --weight_file superpoint_pth_file_path --output_dir superpoint_onnx_file_dir
python convert2onnx/convert_superglue_to_onnx.py --weight_file superglue_pth_file_path --output_dir superglue_onnx_file_dir
```

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
If you modified the image size in the config file, you must delete the old .engine file in the weights dir.
