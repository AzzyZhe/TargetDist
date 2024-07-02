# TargetDetect.py
# 暂且只考虑YOLOv8环境
from ultralytics import YOLO
import torch
import cv2
import numpy as np
# =======================
# 内部配置变量
model_path = 'yolov8n.pt'
# model_path = 'yolov8n.engine'
show_YOLO_details = False
minConf = 0.1
# 外部接口变量 不要在外边改！
use_count = 0
# =======================
model = None

def init():
    print('加载YOLO模型...', end='\n')
    global model
    # 加载模型
    model = YOLO(model_path, task='detect', verbose=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_path.split(sep='.')[-1] == 'pt':
        model.to(device)
    # dummy_image = np.random.random((1, 3, 1080, 1920)).astype(np.float32)
    dummy_image = np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)
    _ = model.predict(source=[dummy_image], verbose=False)
    print(f'完成：{model_path}')

def deal1frame(frame): # 这个封装真的必要吗
    global use_count
    result = model(frame, verbose=show_YOLO_details)[0] # 传入的不是[frame] 只有一个res对象
    # boxes = []
    # for obj in result.boxes:
    #     if float(obj.conf[0]) >= minConf:
    #         boxes.append(obj)
    use_count += 1
    # return boxes
    return result.boxes

# import os
# os.environ['TRT_LOGGER_LEVEL'] = 'WARNING' # tensorRT日志只显示警告
init()
if __name__ == '__main__':
    print('Init Done')
    video = cv2.VideoCapture(0)
    video.set(cv2.CAP_PROP_FRAME_WIDTH,1920)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)
    _, img = video.read()
    video.release()
    results = model.predict(source=img, verbose=True)
    # for res in results:
    #     res.show()
    print('Test Done')
    pass