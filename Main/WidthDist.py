# WidthDist.py
# 写的不标准，没用类封装
import json
import numpy as np
# =======================
# 内部配置变量
pass
# 外部接口变量 不要在外边改！
pass
# =======================
import CameraCalibrate
from ProjUtils import coco_objlist

pixFocal = CameraCalibrate.pixFocal

def dist2Cam(pixWid, Wid):
    return pixFocal * Wid / pixWid # 见CameraCalibrate.py中的注释

def deal1frame(image, boxes):
    # 输入: YOLOv8的result.boxes
    # 输出: [((xywh),cls,conf,dist), ...]
    
    # idx : int = 0
    res = [] # 可绘制的box
    for obj in boxes:
        # idx = idx + 1
        cls = int(obj.cls[0]) 
        if cls >= len(coco_objlist) or cls != coco_objlist[cls].get('label'): # 别的没估，理应按实际确定
            continue
        conf = float(obj.conf[0])
        # 从GPU提取对象位置
        xywh = obj.xywh[0]  # 边界框坐标（中心点+宽高）
        xywh = xywh.cpu().detach()
        xywh = np.float64(xywh)
        xywh = tuple(xywh.tolist())
        # 计算距离
        _, _, pixWid, _ = xywh
        Wid = coco_objlist[cls]['width']
        dist = dist2Cam(pixWid, Wid)
        res.append((xywh, cls, conf, dist))
    return res
    
if __name__ == '__main__':
    pass
