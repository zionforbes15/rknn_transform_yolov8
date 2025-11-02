#以下代码改自https://github.com/rockchip-linux/rknn-toolkit2/tree/master/examples/onnx/yolov5
import cv2
import numpy as np
import pyrealsense2 as rs

OBJ_THRESH, NMS_THRESH, IMG_SIZE = 0.6, 0.4, 640

CLASSES = ("dragon")


def filter_boxes(boxes, box_confidences, box_class_probs):
    """Filter boxes with object threshold.
    """
    box_confidences = box_confidences.reshape(-1)
    candidate, class_num = box_class_probs.shape

    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)

    _class_pos = np.where(class_max_score* box_confidences >= OBJ_THRESH)
    scores = (class_max_score* box_confidences)[_class_pos]

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]

    return boxes, classes, scores

def nms_boxes(boxes, scores):
    """Suppress non-maximal boxes.
    # Returns
        keep: ndarray, index of effective boxes.
    """
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep

# def dfl(position):
#     # Distribution Focal Loss (DFL)
#     import torch
#     x = torch.tensor(position)
#     n,c,h,w = x.shape
#     p_num = 4
#     mc = c//p_num
#     y = x.reshape(n,p_num,mc,h,w)
#     y = y.softmax(2)
#     acc_metrix = torch.tensor(range(mc)).float().reshape(1,1,mc,1,1)
#     y = (y*acc_metrix).sum(2)
#     return y.numpy()

# def dfl(position):
#     # Distribution Focal Loss (DFL)
#     n, c, h, w = position.shape
#     p_num = 4
#     mc = c // p_num
#     y = position.reshape(n, p_num, mc, h, w)
#     exp_y = np.exp(y)
#     y = exp_y / np.sum(exp_y, axis=2, keepdims=True)
#     acc_metrix = np.arange(mc).reshape(1, 1, mc, 1, 1).astype(float)
#     y = (y * acc_metrix).sum(2)
#     return y

def dfl(position):
    # Distribution Focal Loss (DFL)
    # x = np.array(position)
    n,c,h,w = position.shape
    p_num = 4
    mc = c//p_num
    y = position.reshape(n,p_num,mc,h,w)
    
    # Vectorized softmax
    e_y = np.exp(y - np.max(y, axis=2, keepdims=True))  # subtract max for numerical stability
    y = e_y / np.sum(e_y, axis=2, keepdims=True)
    
    acc_metrix = np.arange(mc).reshape(1,1,mc,1,1)
    y = (y*acc_metrix).sum(2)
    return y
    

def box_process(position):
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([IMG_SIZE//grid_h, IMG_SIZE//grid_w]).reshape(1,2,1,1)

    position = dfl(position)
    box_xy  = grid +0.5 -position[:,0:2,:,:]
    box_xy2 = grid +0.5 +position[:,2:4,:,:]
    xyxy = np.concatenate((box_xy*stride, box_xy2*stride), axis=1)

    return xyxy

def yolov8_post_process(input_data):
    boxes, scores, classes_conf = [], [], []
    defualt_branch=3
    pair_per_branch = len(input_data)//defualt_branch
    # Python 忽略 score_sum 输出
    for i in range(defualt_branch):
        boxes.append(box_process(input_data[pair_per_branch*i]))
        classes_conf.append(input_data[pair_per_branch*i+1])
        scores.append(np.ones_like(input_data[pair_per_branch*i+1][:,:1,:,:], dtype=np.float32))

    def sp_flatten(_in):
        ch = _in.shape[1]
        _in = _in.transpose(0,2,3,1)
        return _in.reshape(-1, ch)

    boxes = [sp_flatten(_v) for _v in boxes]
    classes_conf = [sp_flatten(_v) for _v in classes_conf]
    scores = [sp_flatten(_v) for _v in scores]

    boxes = np.concatenate(boxes)
    classes_conf = np.concatenate(classes_conf)
    scores = np.concatenate(scores)

    # filter according to threshold
    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf)

    # nms
    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]
        keep = nms_boxes(b, s)

        if len(keep) != 0:
            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores

def draw(image, boxes, scores, classes, ratio, padding):
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = box
        
        top = (top - padding[0])/ratio[0]
        left = (left - padding[1])/ratio[1]
        right = (right - padding[0])/ratio[0]
        bottom = (bottom - padding[1])/ratio[1]
        # print('class: {}, score: {}'.format(CLASSES[cl], score))
        # print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(top, left, right, bottom))
        top = int(top)
        left = int(left)

        cv2.rectangle(image, (top, left), (int(right), int(bottom)), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                    (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)

def letterbox(im, new_shape=(640, 640), color=(0, 0, 0)):
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - \
        new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right,
                            cv2.BORDER_CONSTANT, value=color)  # add border
    #return im
    return im, ratio, (left, top)

def myFunc(rknn_lite, IMG, raw_depth_frame=None, depth_intrinsics=None):
    """
    返回:
        tuple: (result_image, distance, position_info) 
        其中:
            distance: 检测到的物体距离，如果没有检测到则为None
            position_info: (center_x, center_y, frame_center_x, frame_center_y, deadzone_threshold) 
    """
    original_img = IMG.copy()
    detected_distance = 0.0
    position_info = None
    x_offset_cm=0 
    deadzone_threshold = 6  # 中心区域阈值，单位像素
    
    # 获取帧中心坐标
    frame_center_x = original_img.shape[1] // 2
    frame_center_y = original_img.shape[0] // 2
    
    # 转换颜色空间并缩放
    IMG_RGB = cv2.cvtColor(IMG, cv2.COLOR_BGR2RGB)
    IMG_PROCESSED, ratio, padding = letterbox(IMG_RGB)
    
    # 添加batch维度
    IMG_INPUT = np.expand_dims(IMG_PROCESSED, 0).astype(np.float32)
    
    # 执行推理
    outputs = rknn_lite.inference(inputs=[IMG_INPUT], data_format=['nhwc'])
    
    # 后处理获取检测框
    boxes, classes, scores = yolov8_post_process(outputs)
    
    if boxes is not None and raw_depth_frame is not None and depth_intrinsics is not None:
        for box, score, cls in zip(boxes, scores, classes):
            # 将检测框坐标映射回原始图像尺寸
            x1 = int((box[0] - padding[0]) / ratio[0])
            y1 = int((box[1] - padding[1]) / ratio[1])
            x2 = int((box[2] - padding[0]) / ratio[0])
            y2 = int((box[3] - padding[1]) / ratio[1])
            
            # 计算检测框中心点坐标
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            position_info = (center_x, center_y, frame_center_x, frame_center_y, deadzone_threshold)
            
            # 修改4: 计算像素偏移量
            x_offset_px = center_x - frame_center_x
            

            # 使用深度帧进行精确测距
            try:
                depth = raw_depth_frame.get_distance(center_x, center_y)
                if depth > 0:
                    point_3d = rs.rs2_deproject_pixel_to_point(
                        depth_intrinsics, [center_x, center_y], depth
                    )
                    distance = np.sqrt(point_3d[0]**2 + point_3d[1]**2 + point_3d[2]**2)
                    detected_distance = distance

                      # 修改5: 将像素偏移转换为厘米偏移
                    fx = depth_intrinsics.fx
                    
                    x_offset_cm = (x_offset_px * depth) / fx * 100  # 乘以100转换为厘米
                    
                    #distance_text = f"{distance:.2f}m"
                    distance_text = f"{distance:.2f}m | X:{x_offset_cm:.1f}cm"  # 修改6: 添加偏移量显示
                else:
                    distance_text = "N/A"
            except Exception as e:
                print(f"测距异常: {str(e)}")
                distance_text = "Error"
            
            # 绘制矩形框和文本
            cv2.rectangle(original_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(original_img, (center_x, center_y), 3, (0, 0, 255), -1)
            # 绘制帧中心区域（仅水平方向）
            cv2.line(original_img, 
                   (frame_center_x - deadzone_threshold, frame_center_y - 10),
                   (frame_center_x - deadzone_threshold, frame_center_y + 10),
                   (255, 0, 0), 2)
            cv2.line(original_img, 
                   (frame_center_x + deadzone_threshold, frame_center_y - 10),
                   (frame_center_x + deadzone_threshold, frame_center_y + 10),
                   (255, 0, 0), 2)
            label = f"{CLASSES[cls]} {score:.2f} | {distance_text}"
            cv2.putText(original_img, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # return original_img, detected_distance, position_info
    return original_img, detected_distance, position_info, x_offset_cm # 修改7: 返回offset_cm
