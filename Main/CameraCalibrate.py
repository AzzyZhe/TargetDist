# CameraCalibrate.py
# 写的不标准，没用类封装
import json
import numpy as np
import cv2
# =======================
# 内部配置变量
__check_F = True
camCfg_path = 'CamCfg.json'
default_Device = 1
default_dist = 49.5
square_size = 2.06
# 外部接口变量 不要在外边改！
CamMatrix = None
Distortion = None
pixFocal = None
scale1_T = 1.0
# =======================
# todo: 记录分辨率，不同则重新校准
# 修改scale1_T的处理，暂且太草率了
# =======================
from ProjUtils import ImgUndis

def readCfg(path):
    data = None
    # 从JSON文件加载配置
    with open(path, 'r') as config_file:
        data = json.load(config_file)
    CamMatrix = data.get('CamMatrix')
    if CamMatrix != None:
        CamMatrix = np.array(CamMatrix,dtype=np.float32)
    Distortion = data.get('Distortion')
    if Distortion != None:
        Distortion = np.array(Distortion,dtype=np.float32)
    pixFocal = data.get('pixFocal')
    # global scale1_T
    # scale1_T = pixFocal / default_dist# 1/x?
    return  CamMatrix, Distortion, pixFocal
    
def saveCfg(path):
    data = {'CamMatrix' : CamMatrix.tolist(), 'Distortion' : Distortion.tolist(), 'pixFocal' : pixFocal}
    with open(path, 'w') as config_file:
        json.dump(data, config_file, indent=4)



def reCalibrate(vid_device = default_Device, knownDist = default_dist): # 棋盘格校准内参
    # 打开摄像头
    # 分辨率往上设会自动取可行的
    cap = cv2.VideoCapture(vid_device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)
    pattern_sizes = [(8, 11), (8, 12)]  # 棋盘格板上的内部角点数量
    objpoints = []
    imgpoints = []
    # 失去耐心和美感的暴力写法
    mtx, dist, pfocal = None, None, None
    for pattern_size in pattern_sizes:
        objp = np.zeros((np.prod(pattern_size), 3), dtype=np.float32)
        objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size
        
        _, frame = cap.read()
        # cv2.imshow('debug', frame)
        # cv2.waitKey(0)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        if ret:
            # 找到了格点，从格点校准
            objpoints.append(objp)
            imgpoints.append(corners)
            # print(f'{imgpoints[0][0]}')
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
            # # 去除畸变
            # ImgUndis.init(mtx, dist)
            # img = ImgUndis.deal(frame)
            # cv2.imshow('debug', img)
            # cv2.waitKey(0)
            pixWid0 = np.linalg.norm(imgpoints[0][0] - imgpoints[0][1])
            Dist0 = knownDist
            Wid0 = square_size
            pfocal = pixWid0 * Dist0 / Wid0
            break
        else:
            print(f'尺寸{pattern_size}的棋盘格未找到')
    cap.release()
    # print(f'{objpoints=}, {imgpoints=}')
    # return mtx, dist, objpoints, imgpoints
    return mtx, dist, pfocal
    """
    Dist/Dist0 = (Wid/Wid0)  /          ( pixWid / pixWid0)
               =  Wid/pixWid * (pixWid0 *  Dist0 /    Wid0)
      pixFocal =                pixWid0 *  Dist0 /    Wid0
    """


def main_progress():
    global CamMatrix, Distortion, pixFocal
    # 从JSON文件加载配置
    try:
        CamMatrix, Distortion, pixFocal = readCfg(camCfg_path)
    except Exception as e:
        print(f'读取配置文件失败：{camCfg_path}')
        print(f"错误信息: {e}")
    else:
        print(f'读取配置文件成功：{camCfg_path}')
        
    # 加载不了就重校准
    if any(var is None for var in [CamMatrix, Distortion, pixFocal]):
        # （三个值中有None）
        print( '恢复配置信息失败')
        print(f'从视频设备id={default_Device} 重新校准……')
        
        # 重新校准 CamMatrix, Distortion
        CamMatrix, Distortion, tmpFocal = reCalibrate(default_Device)
        if __check_F:
            # 直接摄像头校准的估计不太准
            pixFocal = tmpFocal
        else:
            print(f'Note: {pixFocal=}，未重新校准')
        
        if any(var is None for var in [CamMatrix, Distortion, pixFocal]):
            print('重新校准失败')
            exit(1)
        
        # 保存配置文件
        try:
            saveCfg(camCfg_path)
            print(f'新内参配置文件存储到了：{camCfg_path}')
        except Exception as e:
            print(f"保存配置文件失败: {e}")
            exit(-1)
            

    

main_progress()
if __name__ == '__main__':
    print(f'{CamMatrix=}')
    print(f'{Distortion=}')
    print(f'{pixFocal=}')