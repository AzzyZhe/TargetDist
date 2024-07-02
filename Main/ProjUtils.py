# ProjUtils.py
import numpy as np
import cv2
import json
# =======================
# 内部配置变量
front_scale = 1.2 # 绘图字体大小系数
objCfg_path = 'objlist.json'
# 外部接口变量 不要在外边改！
pass
# =======================
coco_objlist = None

class MethodNotInitializedException(Exception):
    """Exception raised when a method is called before initialization."""
    def __init__(self, message="Method not initialized"):
        self.message = message
        super().__init__(self.message)

class ImgUndis:
    __camera_matrix = None
    __dist_coeffs = None
    
    @staticmethod
    def init(mtx, dist):
        ImgUndis.__camera_matrix, ImgUndis.__dist_coeffs = mtx, dist
    
    @staticmethod
    def deal(image):
        if ImgUndis.__camera_matrix is None or ImgUndis.__dist_coeffs is None:
            raise MethodNotInitializedException('调用.deal(img)去畸变之前需要先调用.init(mtx,dist)初始化')
        return cv2.undistort(image, ImgUndis.__camera_matrix, ImgUndis.__dist_coeffs)
    
    @classmethod
    def __new__(cls, *args, **kwargs): # 阻止实例化
        raise TypeError("This class cannot be instantiated.")
     
def resizeImage(image, width): # 将图像等比缩放到指定宽度
    (h, w) = image.shape[:2]
    ratio = width / float(w)
    new_height = int(h * ratio)
    # INTER_AREA插值算法一般用于缩小图像
    resized = cv2.resize(image, (width, new_height), interpolation=cv2.INTER_LINEAR)
    return resized

class DistDraw:
    # __objLabels = None
    # @staticmethod
    # def init(cfgPath):
    #     with open(cfgPath, 'r') as config_file:
    #         data = json.load(config_file)
    #     objLabels = []
    #     for item in data:
    #         if item != {}:
    #             objLabels.append(item['label_name'])
    #         else:
    #             objLabels.append(None)
    #     DistDraw.__objLabels = objLabels
    #     # print(DistDraw.__objLabels)
    
    @staticmethod
    def deal1frame(image, dat):
        # dat: [((xywh),cls,conf,dist), ...]
        # if DistDraw.__objLabels is None:
        #     raise MethodNotInitializedException('调用.deal(img,dat)绘制距离之前需要先调用.init()初始化')
        for xywh, cls, conf, dist in dat:
            cls_label = coco_objlist[cls].get('label_name')
            # 目标位置从xywh转化为四点的box形式
            bx, by, bw, bh = xywh
            box_center = (bx, by)
            box_size = (bw, bh)
            box = (box_center, box_size, 0)
            box = cv2.boxPoints(box)
            box = np.intp(box)
            # 绘制方框、类别、距离在图中
            cv2.drawContours(image, [box], -1, (0, 0, 255), 2)
            label_pos = box[1] + [0, int(20*front_scale)]
            cv2.putText(image, "%s" % cls_label,
                        label_pos, cv2.FONT_HERSHEY_SIMPLEX,
                        front_scale, (0, 255, 0), 3)
            dist_pos = box[0]
            cv2.putText(image, "%.2fcm" % dist,
                        dist_pos, cv2.FONT_HERSHEY_SIMPLEX,
                        front_scale, (0, 255, 0), 3)
        return image
    
    @classmethod
    def __new__(cls, *args, **kwargs): # 阻止实例化
        raise TypeError("This class cannot be instantiated.")

def readCfg(path):
    data = None
    # 从JSON文件加载配置
    with open(path, 'r') as config_file:
        data = json.load(config_file)
    return data

def init():
    global coco_objlist
    coco_objlist = readCfg(objCfg_path) # 不额外判错了
    print(f'读取对象列表完成: {objCfg_path}')

init()
if __name__ == '__main__':
    # print(DistDraw.__objLabels)
    # video = cv2.VideoCapture(1)
    # video.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
    # video.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
    # _, img = video.read()
    # video.release()
    pass