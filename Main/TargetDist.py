# main.py
# 目前没处理画面畸变
import cv2
import numpy as np
import sys
from datetime import datetime
import time
# =======================
# 内部配置变量
## 基础设置
# default_vidDevice = 0
default_vidDevice = R"./vid3-cars.mp4"
useTriangleDist = False
# useTriangleDist = True
frame_sep = 1
## 屏幕显示
displayOnScreen = False
# displayOnScreen = True
dispWid = 1080
## 视频保存 
saveVid = False
outputFps = 30.06
outputPath = './Output_%s.mp4' % datetime.now().strftime('%Y%m%d_%H%M%S')
# =======================
# 目前测试下保存视频有速度（帧数）控制不对的问题
# 时间有限，不是核心功能暂且忽略吧
# =======================
from ProjUtils import DistDraw, resizeImage
import CameraCalibrate
import TargetDetect
if useTriangleDist:
    import TriangleDist as Dist
else:
    import WidthDist as Dist

def deal1frame(frame):
    boxes = TargetDetect.deal1frame(frame)
    ret = Dist.deal1frame(frame, boxes)
    return ret

def main():
    print('本次视频输入设备:', default_vidDevice)
    cap = cv2.VideoCapture(default_vidDevice, cv2.CAP_V4L if type(default_vidDevice) == int else cv2.CAP_ANY)  # 初始化一个OpenCV的视频读取对象
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,10000) # 初始化
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,10000)
    if saveVid:
        print('本次运行将尝试记录视频')
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(outputPath, fourcc, outputFps, (frame_width, frame_height))
    print('本次使用测距方法:', 'TraingleDist' if useTriangleDist else 'WidthDist')
    
    print('开始运行...')
    startTime = time.time()
    tot_count = 0
    exe_count = 0
    try:
        while True:
            _, frame = cap.read()
            if frame is None:
                print('视频流获取完成')
                break
            tot_count += 1
            if tot_count % frame_sep != 0:
                continue
            exe_count += 1
            boxes = TargetDetect.deal1frame(frame)
            ret = Dist.deal1frame(frame, boxes)
            if displayOnScreen or saveVid:
                img = DistDraw.deal1frame(frame, ret)
                if displayOnScreen:
                    cv2.imshow('DistDetect', resizeImage(img, dispWid))
                    cv2.waitKey(20)
                    # cv2.waitKey(0)
                if saveVid:
                    video_writer.write(img)
    except KeyboardInterrupt:
        print('运行中按键中断')
    # except Exception as e:
    #     print(f"主函数运行错误: {e}")
    finally:
        print('正在退出...')
        endTime = time.time()
        cap.release()
        if saveVid:
            video_writer.release()
            print(f'已保存视频：{outputPath}')
        cv2.destroyAllWindows()
    execTime = endTime - startTime
    print(f'执行用时: {execTime:.6f}s')
    print(f'共处理了 {exe_count} 帧')
    # 这个好像不太准确
    print(f'fps: {exe_count / execTime:.6f}')

if __name__ == '__main__':
    main()
else:
    pass