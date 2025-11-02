# rknn_transform_yolov8



Using the Orange Pi 5 Max development board as the host computer, the system deploys the YOLOv8 visual model for inference. The process involves converting the .pt file to ONNX format, then quantizing and pruning the ONNX file to generate an INT8-format RKNN file. Leveraging the onboard RK3588 chip’s NPU for accelerated inference, the system integrates an Intel D455 depth camera to achieve real-time object detection and depth measurement. The measured distance is evaluated against a predefined range, and specific data is transmitted to the lower-level device via UART serial communication, enabling real-time data transfer.Finally, real-time monitoring is achieved by streaming to an HTML website.

利用香橙派5max开发板作为上位机，搭载yolov8视觉模型推理，实现.pt文件转换.onnx,再由onnx文件量化剪枝转换为int8格式的rknn文件，最后利用板载的rk3588芯片的NPU加速推理，结合Intel D455深度相机，实现对特定物体的实时检测与深度测距，并将距离带入范围检测，发送特定数据到下位机的uart串口，实现实时通信传输。最后通过推流至html网站实时监测.



This is my first attempt at related development as a beginner. I modified and built the training system using the relevant source code. For detailed model training and transformation information, please see:

https://github.com/airockchip/ultralytics_yolov8

https://github.com/airockchip/rknn_model_zoo

https://github.com/airockchip/rknn-toolkit2

Specific process reference:

https://blog.csdn.net/weixin_45686120/article/details/143362531?spm=1001.2014.3001.5506

https://blog.csdn.net/qq_42541521/article/details/139603988?spm=1001.2014.3001.5506



本人小白第一次进行相关开发尝试，利用相关源码进行修改构建训练，详模型训练转化详情可见：

https://github.com/airockchip/ultralytics_yolov8

https://github.com/airockchip/rknn_model_zoo

https://github.com/airockchip/rknn-toolkit2

具体流程参考：

https://blog.csdn.net/weixin_45686120/article/details/143362531?spm=1001.2014.3001.5506

https://blog.csdn.net/qq_42541521/article/details/139603988?spm=1001.2014.3001.5506

