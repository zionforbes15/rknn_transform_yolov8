import cv2
import time
import numpy as np
import pyrealsense2 as rs
import serial
from rknnpool import rknnPoolExecutor
from func import myFunc

class DepthCamera:
    def __init__(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.disable_stream(rs.stream.accel)
        config.disable_stream(rs.stream.gyro)
        config.disable_stream(rs.stream.infrared)
        config.enable_stream(rs.stream.depth,480,270,rs.format.z16,30)
        config.enable_stream(rs.stream.color,640,360,rs.format.bgr8,30)
        try:
            self.profile=self.pipeline.start(config)
            depth_sensor=self.profile.get_device().first_depth_sensor()
            color_sensor=self.profile.get_device().first_color_sensor()
            depth_sensor.set_option(rs.option.enable_auto_exposure,0)
            depth_sensor.set_option(rs.option.exposure,3000)
            depth_sensor.set_option(rs.option.emitter_enabled,0)
            color_sensor.set_option(rs.option.enable_auto_white_balance,1)
            color_sensor.set_option(rs.option.enable_auto_exposure,1)
            self.depth_scale=depth_sensor.get_depth_scale()
            depth_profile=self.profile.get_stream(rs.stream.depth).as_video_stream_profile()
            self.depth_intrinsics=depth_profile.get_intrinsics()
            self.align=rs.align(rs.stream.color)
            print("D455相机初始化完成")
        except Exception as e:
            print(f"相机初始化失败:{str(e)}")
            raise RuntimeError("无法启动相机设备")

    def get_frame(self):
        try:
            frames=self.pipeline.wait_for_frames(timeout_ms=100)
            aligned_frames=self.align.process(frames)
            raw_depth_frame=aligned_frames.get_depth_frame()
            color_frame=aligned_frames.get_color_frame()
            if not raw_depth_frame or not color_frame:
                return False,None,None,None
            return True,np.asanyarray(color_frame.get_data()),\
                   np.asanyarray(raw_depth_frame.get_data()),raw_depth_frame
        except Exception as e:
            print(f"帧获取异常:{str(e)}")
            return False,None,None,None

    def release(self):
        if hasattr(self,'pipeline'):
            self.pipeline.stop()
            print("相机资源已释放")

class UartHandler:
    def __init__(self,port='/dev/ttyUSB0',baudrate=115200):
        try:
            self.ser=serial.Serial(
                port=port,
                baudrate=baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.01
            )
            self.last_send_time=time.monotonic()
            print(f"串口{port}初始化成功")
        except Exception as e:
            print(f"串口初始化失败:{str(e)}")
            self.ser=None

    def send_packet(self,distance_cmd,position_cmd):
        if not self.ser or not self.ser.is_open:
            return False
        try:
            packet=bytearray()
            packet.append(0xFF)
            packet.append(ord(str(distance_cmd)))
            packet.append(ord(position_cmd.lower()))
            packet.append(0xFE)
            self.ser.write(packet)
            return True
        except Exception as e:
            print(f"串口发送失败:{str(e)}")
            return False

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("串口已关闭")

def main():
    try:
        dc=DepthCamera()
        uart=UartHandler(port='/dev/ttyUSB0',baudrate=115200)
        pool=rknnPoolExecutor(
            rknnModel="/home/orangepi/rknn3588-yolov8/rknnModel/bestR1.rknn",
            TPEs=3,
            func=myFunc
        )
        for _ in range(4):
            ret,color,_,raw_depth_frame=dc.get_frame()
            if not ret:
                dc.release()
                pool.release()
                exit(-1)
            pool.put(color)

        frames,loopTime,initTime=0,time.time(),time.time()
        current_cmd='0'
        position_cmd='8'
        try:
            while True:
                start_time=time.time()
                frames+=1
                ret,bgr_frame,_,raw_depth_frame=dc.get_frame()
                if not ret:continue
                result_img,detected_distance,position_info=myFunc(pool.rknnPool[0],bgr_frame,
                                                    raw_depth_frame=raw_depth_frame,
                                                    depth_intrinsics=dc.depth_intrinsics)
                if detected_distance is not None:
                    if 1.0<=detected_distance<1.4:current_cmd='1'
                    elif 1.4<=detected_distance<1.8:current_cmd='2'
                    elif 1.8<=detected_distance<2.2:current_cmd='3'
                    elif 2.2<=detected_distance<2.6:current_cmd='4'
                    elif 2.6<=detected_distance<=3.0:current_cmd='5'
                    else:current_cmd='0'
                    if position_info is not None:
                        cx,_,frame_center_x,_,deadzone=position_info
                        if cx<frame_center_x-deadzone:position_cmd='6'
                        elif cx>frame_center_x+deadzone:position_cmd='7'
                        else:position_cmd='8'
                else:
                    current_cmd='0'
                    position_cmd='8'

                if (time.monotonic()-uart.last_send_time)>=0.1:
                    debug_str=f"[{time.strftime('%H:%M:%S')}]"
                    if uart.send_packet(current_cmd,position_cmd):
                        packet_data=f"FF{current_cmd}{position_cmd}FE"
                        debug_str+=f"发送:{packet_data}"
                    if detected_distance is not None:
                        debug_str+=f"距离:{detected_distance:.2f}m"
                    print(debug_str)
                    uart.last_send_time=time.monotonic()
                
                fps=1/(time.time()-start_time)
                cv2.putText(result_img,f"CMD:{current_cmd}{position_cmd}|FPS:{fps:.1f}", 
                           (10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
                cv2.imshow("D455ObjectDetectionwithDepth",result_img)
                
                if frames%30==0:
                    avg_fps=30/(time.time()-loopTime)
                    print(f"30帧平均帧率:{avg_fps:.1f}FPS")
                    loopTime=time.time()
                
                if cv2.waitKey(1)==27:break

        finally:
            dc.release()
            uart.close()
            pool.release()
            cv2.destroyAllWindows()
            print(f"总运行帧数:{frames}")
            print(f"总平均帧率:{frames/(time.time()-initTime):.1f}FPS")
    except Exception as e:
        print(f"设备初始化失败:{e}")

if __name__=="__main__":
    main()