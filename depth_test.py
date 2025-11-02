import cv2
import time
import numpy as np
import pyrealsense2 as rs
from rknnpool import rknnPoolExecutor
from func import myFunc  # 确保myFunc已更新为带深度测距的版本

class DepthCamera:
    def __init__(self):
        # 初始化RealSense相机
        self.pipeline = rs.pipeline()
        config = rs.config()
        
        # 禁用所有非必要模块
        config.disable_stream(rs.stream.accel)  # 禁用加速度计
        config.disable_stream(rs.stream.gyro)   # 禁用陀螺仪
        config.disable_stream(rs.stream.infrared)  # 禁用红外流
        
        # 配置深度和彩色流
        config.enable_stream(rs.stream.depth, 480, 270, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 360, rs.format.bgr8, 30)
        
        # 启动设备
        try:
            self.profile = self.pipeline.start(config)
            
            # 获取传感器并优化设置
            depth_sensor = self.profile.get_device().first_depth_sensor()
            color_sensor = self.profile.get_device().first_color_sensor()
            
            # 深度传感器设置
            depth_sensor.set_option(rs.option.enable_auto_exposure, 0)
            depth_sensor.set_option(rs.option.exposure, 3000)
            depth_sensor.set_option(rs.option.emitter_enabled, 0)  # 关闭红外发射器
            
            # 彩色传感器设置
            color_sensor.set_option(rs.option.enable_auto_white_balance, 1)
            color_sensor.set_option(rs.option.enable_auto_exposure, 1)
            
            # 获取深度参数
            self.depth_scale = depth_sensor.get_depth_scale()
            depth_profile = self.profile.get_stream(rs.stream.depth).as_video_stream_profile()
            self.depth_intrinsics = depth_profile.get_intrinsics()
            
            # 初始化对齐器
            self.align = rs.align(rs.stream.color)
            
            print("D455相机初始化完成")
            
        except Exception as e:
            print(f"相机初始化失败: {str(e)}")
            raise RuntimeError("无法启动相机设备")

    def get_frame(self):
        """获取对齐后的帧数据"""
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=100)
            aligned_frames = self.align.process(frames)
            
            raw_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not raw_depth_frame or not color_frame:
                return False, None, None, None

            return True, np.asanyarray(color_frame.get_data()), \
                   np.asanyarray(raw_depth_frame.get_data()), raw_depth_frame
                   
        except Exception as e:
            print(f"帧获取异常: {str(e)}")
            return False, None, None, None

    def release(self):
        """释放资源"""
        if hasattr(self, 'pipeline'):
            self.pipeline.stop()
            print("相机资源已释放")

def main():
    # 初始化设备
    try:
        dc = DepthCamera()
    except Exception as e:
        print(f"设备初始化失败: {e}")
        return
    
    # 初始化RKNN线程池
    modelPath = "./rknnModel/bestR1.rknn"
    TPEs = 3  # 线程数
    pool = rknnPoolExecutor(
        rknnModel=modelPath,
        TPEs=TPEs,
        func=myFunc
    )
    
    # 初始化缓冲帧
    for _ in range(TPEs + 1):
        ret, color, _, raw_depth_frame = dc.get_frame()
        if not ret:
            dc.release()
            pool.release()
            exit(-1)
        pool.put(color)

    frames, loopTime, initTime = 0, time.time(), time.time()
    
    try:
        while True:
            start_time = time.time()
            frames += 1
            
            # 获取帧数据（包含原始深度帧）
            ret, bgr_frame, _, raw_depth_frame = dc.get_frame()
            if not ret:
                continue
            
            # 提交处理请求（传递深度信息）
            result = myFunc(pool.rknnPool[0], bgr_frame,
                          raw_depth_frame=raw_depth_frame,
                          depth_intrinsics=dc.depth_intrinsics)
            
            # 显示处理结果
            cv2.imshow("D455 Object Detection with Depth", result)
            
            # 性能日志（每30帧）
            if frames % 30 == 0:
                avg_fps = 30 / (time.time() - loopTime)
                print(f"30帧平均帧率: {avg_fps:.1f}FPS")
                loopTime = time.time()
            
            if cv2.waitKey(1) == 27:  # ESC键退出
                break

    finally:
        # 确保资源释放
        dc.release()
        pool.release()
        cv2.destroyAllWindows()
        print(f"总运行帧数: {frames}")
        print(f"总平均帧率: {frames/(time.time()-initTime):.1f}FPS")

if __name__ == "__main__":
    main()