# CasNet - TensorFlow-API
## Semantic segmentation multi-task with TensorFlow API 2.x

The [CasNet](https://ieeexplore.ieee.org/document/7873262) architecture is a cascaded end-to-end convolutional neural network proposed for two simultaneosly tasks: road detection and centerline extraction. The overal architecture of that model is shown below.

![](images/casnet_img.png)

The model has a input (RGB images) and two outputs (road and centerline). The model was build to use with data loading pipeline of TensorFlow API.
