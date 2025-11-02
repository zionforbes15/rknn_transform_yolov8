import cv2
import time
import numpy as np
import threading
import pyrealsense2 as rs
import serial  # 新增串口库
import pytesseract
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

# class UartHandler:
#     def __init__(self, port='/dev/ttyUSB0', baudrate=115200):
#         """初始化串口通信"""
#         try:
#             self.ser = serial.Serial(
#                 port=port,
#                 baudrate=baudrate,
#                 bytesize=serial.EIGHTBITS,
#                 parity=serial.PARITY_NONE,
#                 stopbits=serial.STOPBITS_ONE,
#                 timeout=0.01
#             )
#             self.last_send_time = time.monotonic()
#             print(f"串口 {port} 初始化成功")
#         except Exception as e:
#             print(f"串口初始化失败: {str(e)}")
#             self.ser = None

#     def send_command(self, cmd):
#         if isinstance(cmd, str):
#             cmd = cmd.encode()  # 确保是 bytes
#         self.ser.write(cmd)  # 直接写入 bytes
#         return True

#     def close(self):
#         """关闭串口"""
#         if self.ser and self.ser.is_open:
#             self.ser.close()
#             print("串口已关闭")

class UartHandler(threading.Thread):
    def __init__(self, com="/dev/ttyUSB0", baudrate=115200, timeout=1):
        threading.Thread.__init__(self)
        self.serial_port = com
        self.baudrate = baudrate
        self.timeout = timeout
        self.running = True
        self.serial_conn = None
        self.data_queue = None  # 用于存储要发送的数据
        self.lock = threading.Lock()  # 线程安全锁
        self.data_available = threading.Event()  # 数据可用事件

    def stop(self):
        """停止线程"""
        self.running = False
        self.join()  # 等待线程结束

    def uart_send_command(self, task_id, param, wait=True, timeout=1):
        print(f"@{task_id:02d}!{param:02d}#")
        self.serial_conn.write(f"@{task_id:02d}!{param:02d}#".encode("utf-8"))
        #time.sleep(0.1)  # 避免单片机清零不及时
        if wait is True:
            self.wait_for_32_ack(timeout, task_id)

    def wait_for_32_ack(self, timeout=1, *args):
        time_begin = time.time()
        time_end = time_begin + timeout
        task = list(args)
        while task:
            if time.time() >= time_end:
                print("time out")
                break
            if self.serial_conn.inWaiting() >= 2:
                rx_buf = self.serial_conn.read(2).decode("utf-8")
                ack = int(rx_buf)
                if ack in task:
                    print(f"get  {ack}")
                    task.remove(ack)
            else:
                time.sleep(0.02)
        print(f"get ack {args}")

    def add_data(self, data):
        with self.lock:
            self.data_queue = data  # 直接替换为最新数据
            self.data_available.set()  # 触发数据可用事件
    def run(self):
        try:
            self.serial_conn = serial.Serial(
                port=self.serial_port,
                baudrate=self.baudrate,
                timeout=self.timeout
            )
            print(f"串口 {self.serial_port} 已打开")
            
            while self.running:
                self.data_available.wait(timeout=1)
                
                # 检查是否有数据
                with self.lock:
                    if self.data_queue is not None:
                        data = self.data_queue
                        self.data_queue = None  # 清空队列
                        try:
                            if isinstance(data, str):
                                data = data.encode('utf-8')
                            self.uart_send_command(task_id=0, param=data, wait=False)
                        except Exception as e:
                            print(f"发送数据时出错: {e}")
                
                # 如果数据已处理，清除事件
                with self.lock:
                    if self.data_queue is None:
                        self.data_available.clear()
            
        except serial.SerialException as e:
            print(f"无法打开串口 {self.serial_port}: {e}")
        finally:
            if self.serial_conn and self.serial_conn.is_open:
                self.serial_conn.close()
                print("串口已关闭")


def main():
    # 初始化设备
    try:
        dc = DepthCamera()
        
        uart = UartHandler(com ='/dev/ttyUSB1', baudrate=115200)
        uart.start()
    except Exception as e:
        print(f"设备初始化失败: {e}")
        return
    
    # 初始化RKNN线程池
    modelPath = "/home/orangepi/rknn3588-yolov8/rknnModel/bestR1.rknn"
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
    current_cmd = 0  # 默认命令
    position_cmd = 0  # 位置命令
    
    try:
        while True:
            start_time = time.time()
            frames += 1
            
            # 获取帧数据
            ret, bgr_frame, _, raw_depth_frame = dc.get_frame()
            if not ret:
                continue
            
            # 提交处理请求并获取结果
            result_img, detected_distance, position_info, offset_cm= myFunc(pool.rknnPool[0], bgr_frame,
                                                raw_depth_frame=raw_depth_frame,
                                                depth_intrinsics=dc.depth_intrinsics)
            
              # 修改2: 添加偏移量处理
            x_offset_cm, y_offset_cm = offset_cm
            
            # 根据距离确定控制命令
            if detected_distance is not None:
                if 1.0 <= detected_distance < 1.4:
                    current_cmd = 1
                elif 1.4 <= detected_distance < 1.8:
                    current_cmd = 2
                elif 1.8 <= detected_distance < 2.2:
                    current_cmd = 3
                elif 2.2 <= detected_distance < 2.6:
                    current_cmd = 4
                elif 2.6 <= detected_distance <= 3.0:
                    current_cmd = 5
                else:
                    current_cmd = 0
                
                # 根据位置信息确定位置命令
                if position_info is not None:
                    center_x, center_y, frame_center_x, frame_center_y, deadzone = position_info
                    if center_x < frame_center_x - deadzone:  # 左边
                        position_cmd = 6
                    elif center_x > frame_center_x + deadzone:  # 右边
                        position_cmd = 7
                    else:
                        position_cmd = 8  # 中心区域
            else:
                current_cmd = 0
                position_cmd = 0

            # 串口通信部分（严格20ms发送周期控制）
                # if (time.monotonic() - uart.last_send_time) >= 0.02:
                #     debug_str = f"[{time.strftime('%H:%M:%S')}] "
                
                # # 优先发送位置命令
                # if position_cmd:
                #     if uart.send_command(position_cmd.encode()):
                #         debug_str += f"位置指令: {position_cmd} "
                # # 如果没有位置命令，发送距离命令
                # elif current_cmd != 0:
                #     if uart.send_command(current_cmd):
            #     #         debug_str += f"距离指令: {current_cmd} "

            data_to_send = current_cmd*10+position_cmd
            uart.add_data(data_to_send)
            # if time_flag:
            #     time_flag = False
            #     time_begin = time.time()
            #     time_end = time_begin + 0.1
            # if time.time()>time_end:
            #     uart.uart_send_command(task_id=0, param = (current_cmd*10+position_cmd), wait=False)
            #     time_begin = time.time()
            #     time_end = time_begin + 0.1
            #     # if detected_distance is not None:
            #     #     debug_str = f"距离: {detected_distance:.2f}m"
            #     #     print(debug_str)
            
            # 在图像上显示信息
            fps = 1 / (time.time() - start_time)
            cv2.putText(result_img, f"CMD: {current_cmd} | POS: {position_cmd} | FPS: {fps:.1f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 显示处理结果
            cv2.imshow("D455 Object Detection with Depth", result_img)
            
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
        uart.close()
        pool.release()
        cv2.destroyAllWindows()
        print(f"总运行帧数: {frames}")
        print(f"总平均帧率: {frames/(time.time()-initTime):.1f}FPS")
        
if __name__ == "__main__":
    main()