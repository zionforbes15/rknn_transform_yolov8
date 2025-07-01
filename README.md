# rknn_transform_yolov8
The Orange Pi 5 Max serves as the host computer, running YOLOv8 for real-time object detection. The model is converted from .pt to ONNX, then quantized and pruned into an INT8 RKNN file for efficient NPU acceleration on the RK3588 chip.
