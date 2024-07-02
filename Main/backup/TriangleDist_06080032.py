# TriangleDist.py
# 写的不标准，没用类封装
# WIP
import json
import cv2
import numpy as np
# =======================
# 内部配置变量
scale2_T = 1.0 # 二次暴力校准
# 外部接口变量 不要在外边改！
pass
# =======================
import CameraCalibrate
from ProjUtils import resizeImage, coco_objlist

pixFocal = CameraCalibrate.pixFocal
K = CameraCalibrate.CamMatrix
scale_T = CameraCalibrate.scale1_T * scale2_T

def getProjectionMatrix(K, pose_matrix):
    # 从位姿矩阵中提取旋转和平移部分构建投影矩阵
    R = pose_matrix[:3, :3]
    t = pose_matrix[:3, 3]
    return np.dot(K, np.hstack((R, t.reshape(3, 1)))), t
def assemblePose(rotation, translation):
    # 构建位姿矩阵
    pose_matrix = np.eye(4)
    pose_matrix[:3, :3] = rotation
    pose_matrix[:3, 3] = translation.flatten()
    return pose_matrix

# 此函数可能不用于正式代码
def compute_depth_and_projection(world_point_homogeneous, camera_pose_matrix, camera_matrix):
    # 将世界坐标转换为齐次坐标
    # world_point_homogeneous = np.array([world_point[0], world_point[1], world_point[2], 1])
    # 使用相机位姿矩阵将点从世界坐标系转换到相机坐标系
    camera_point_homogeneous = np.dot(camera_pose_matrix, world_point_homogeneous)
    # 计算点在相机坐标系中的深度
    depth = camera_point_homogeneous[2]
    # 从齐次坐标中提取非齐次坐标
    camera_point = camera_point_homogeneous[:3]
    # 使用相机内参矩阵将点投影到相机图像上
    projected_point_homogeneous = np.dot(camera_matrix, camera_point)
    projected_point = projected_point_homogeneous[:2] / projected_point_homogeneous[2]
    return depth, projected_point

def getMatchesAndPose(image1, image2, cameraMatrix):
    if image1 is None or image2 is None:
        print("Error: Could not read images.")
        return None, None, None

    # 初始化特征检测器和描述子提取器
    # detector = cv2.ORB_create()
    detector = cv2.BRISK_create()
    # detector = cv2.AKAZE_create()
    # extractor = cv2.ORB_create()
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, None)
    # 检测关键点和计算描述子
    keypoints1, descriptors1 = detector.detectAndCompute(image1, None)
    keypoints2, descriptors2 = detector.detectAndCompute(image2, None)

    # 创建匹配器
    matches = matcher.match(descriptors1, descriptors2)

    # 筛选匹配点
    good_matches = []
    for match in matches:
        if match.distance < 50:  # 设置一个匹配距离阈值
            good_matches.append(match)
    # print(good_matches)

    # 计算相机位姿变化
    if len(good_matches) > 4:
        src_pts = np.float32([keypoints1[match.queryIdx].pt for match in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[match.trainIdx].pt for match in good_matches]).reshape(-1, 1, 2)
        print(f'{src_pts=}')
        # 计算基础矩阵
        fundamental_matrix, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC)
        mask = mask.flatten().astype(bool)
        src_pts = src_pts[mask]
        dst_pts = dst_pts[mask]

        # 计算旋转和平移向量
        retval, rotation_vector, translation_vector, mask = cv2.recoverPose(fundamental_matrix, src_pts, dst_pts, cameraMatrix)

        # 输出结果
        # print("Rotation:", rotation_vector)
        # print("Translation:", translation_vector)
        
        matched_points = [[src_pt.flatten().tolist(), dst_pt.flatten().tolist()] for src_pt, dst_pt in zip(src_pts, dst_pts)]
        # mask = mask.flatten().astype(bool)
        # matched_points = matched_points[mask]
        matched_points = np.array(matched_points,dtype=np.float32)
        # return good_matches, rotation_vector, translation_vector
        return matched_points, rotation_vector, translation_vector
    else:
        print("Not enough good matches found - {}/{}".format(len(good_matches), len(matches)))
        return None, None, None



def recover2Boxes(boxes, pts):
    # 输入: boxes: [(xywh), ...] pts: [((xy), dist), ...] 
    # 输出: [dist, ...] 顺序同 boxes
    # 暴力匹配 O(NM)
    ret = [] 
    for x, y, w, h in boxes:
        xy1 = (x - w / 2, y - h / 2)
        xy2 = (x + w / 2, y + h / 2)
        cnt, tot = 0
        for pos, dis in pts:
            if xy1[0] <= pos[0] <= xy2[0] and xy1[1] <= pos[1] <= xy2[1]: # 链式比较在Python中可行
                cnt += 1
                tot += dis
        if cnt > 0:
            ret.append(tot / cnt)
        else:
            ret.append(np.NaN)
    return ret

def getPointDistances(last_frame, now_frame):
    points, rotation_vector, translation_vector = getMatchesAndPose(last_frame, now_frame, K)
    if points is None:
        print('Failed!')
        return None, None
    # print(f'{points=}')
    print(f'{translation_vector=}')
    len_t = np.linalg.norm(translation_vector)
    print(f'{len_t=}')
    # 相机位姿矩阵
    Pose0 = np.array(np.eye(4),dtype=np.float32)
    Pose1 = assemblePose(rotation_vector, translation_vector)
    P0, pCam_0 = getProjectionMatrix(K,Pose0)
    P1, pCam_1 = getProjectionMatrix(K,Pose1)
    point_seq = 5
    cnt = 0
    pts, dists = [], []
    for p in points:
        cnt = cnt + 1
        if cnt % point_seq != 0:
            continue
            
        # print('2D coordinates: {}, {}'.format(p[0], p[1]))
        s = np.array(cv2.triangulatePoints(P0, P1, 
                                        p[0], 
                                        p[1])).T
        # print(f'{s=}')
        # p_3d_homo=s[0]
        p_3d_homo=s[0]/s[0][-1]
        p_3d=s[0][:-1]/s[0][-1]
        dis0 = np.linalg.norm(p_3d - pCam_0)
        dis1 = np.linalg.norm(p_3d - pCam_1)
        dist = (dis0+dis1)/2
        pt = p[1]
        pts.append(pt)
        dists.append(dist)
    return pts, dists
    
    

def init():
    pass

last_image = None
def deal1frame(image, boxes):
    # 输入: boxes: YOLOv8的result.boxes
    # 输出: [((xywh),cls,conf,dist), ...]
    global last_image
    # idx = 0
    pboxes = []
    tmp = []
    for obj in boxes:
        # idx = idx + 1
        # 此重复部分或应存为函数？
        cls = int(obj.cls[0]) 
        if cls >= len(coco_objlist) or cls != coco_objlist[cls].get('label'): # 别的没估，按实际确定
            continue
        conf = float(obj.conf[0])
        # 从GPU提取对象位置
        xywh = obj.xywh[0]  # 边界框坐标（中心点+宽高）
        xywh = xywh.cpu().detach()
        xywh = np.float64(xywh)
        xywh = tuple(xywh.tolist())
        pboxes.append(xywh)
        tmp.append((cls,conf))
    
    ret = []
    
    last_image = image
    return ret

def drawDistPts(img, pts, dists, point_sep = 5):
    cnt = 0
    for pt, dist in zip(pts, dists):
        cnt = cnt + 1
        if cnt % point_sep != 0:
            continue
        center = (pt[0],pt[1])
        center = np.intp(center)
        # 绘制点
        cv2.circle(img, center, 5, (0, 0, 255), -1)
        # 在点旁边标注距离
        text_pos = tuple(pt + 10)  # 文字位置的微小偏移
        text_pos = np.intp(text_pos)
        # print(f"{text_pos=}")
        cv2.putText(img, f"{dist:.2f}", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img

def test1():
    from ProjUtils import DistDraw
    point_seq = 5
    frame_seq = 20
    debug_device = R"VID_20240606_215815.mp4"
    cap = cv2.VideoCapture(debug_device)
    frame_count = 0
    process_count = 0
    ret, now_frame = cap.read()

    try:
        while True:
            last_frame = now_frame
            ret, now_frame = cap.read()
            # now_frame = cv2.cvtColor(now_frame, cv2.COLOR_BGR2GRAY)
            if not ret:
                break
            frame_count += 1
            if frame_count % frame_seq != 0:
                continue
            
            process_count += 1
            print(f'Frame{process_count}...')
            
            points, rotation_vector, translation_vector = getMatchesAndPose(last_frame, now_frame, K)
            if points is None:
                print('Failed!')
                continue
            # print(f'{points=}')
            print(f'{translation_vector=}')
            len_t = np.linalg.norm(translation_vector)
            print(f'{len_t=}')

            # 相机位姿矩阵
            Pose0 = np.array(np.eye(4),dtype=np.float32)
            Pose1 = assemblePose(rotation_vector, translation_vector)

            P0, pCam_0 = getProjectionMatrix(K,Pose0)
            P1, pCam_1 = getProjectionMatrix(K,Pose1)

            # print('Projection matrices:')
            # print(P0,'\n',P1, '\n')

            img = now_frame.copy()
            cnt = 0
            for p in points:
                cnt = cnt + 1
                if cnt % point_seq != 0:
                    continue
                
                # print('2D coordinates: {}, {}'.format(p[0], p[1]))
                s = np.array(cv2.triangulatePoints(P0, P1, 
                                        p[0], 
                                        p[1])).T
                # print(f'{s=}')
                # p_3d_homo=s[0]
                p_3d_homo=s[0]/s[0][-1]
                p_3d=s[0][:-1]/s[0][-1]
                # print(p_3d_homo)
                # print(p_3d_homo.T)
                # print('3D coordinates: {}'.format(p_3d))
                
                dis0 = np.linalg.norm(p_3d - pCam_0)
                dis1 = np.linalg.norm(p_3d - pCam_1)
                # print(f'{p_3d=}')
                # print(f'{dis0=} {dis1=}')
                # depth, projected_point = compute_depth_and_projection(p_3d_homo, Pose0, K)
                # print(f'{depth=} {projected_point=}')
                # depth, projected_point = compute_depth_and_projection(p_3d_homo, Pose1, K)
                # print(f'{depth=} {projected_point=}')
                
                # dist = -depth
                dist = (dis0+dis1)/2
                # pt = principal_point
                pt = p[1]
                # print(f"{pt=} {dist=}")
                
                center = (pt[0],pt[1])
                center = np.intp(center)
                # print(f"{center=}")
                cv2.circle(img, center, 5, (0, 0, 255), -1)
                # 在点旁边标注距离
                text_pos = tuple(pt + 10)  # 文字位置的微小偏移
                text_pos = np.intp(text_pos)
                # print(f"{text_pos=}")
                cv2.putText(img, f"{dist:.2f}", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            print(f"Ponit{cnt=}")
            # 显示图像
            img = resizeImage(img, 1080)
            cv2.imshow('Image with Points and Distances', img)
            cv2.waitKey(0)
    except KeyboardInterrupt:
        print('Interrupted')
    finally:
        cv2.destroyAllWindows()
        cap.release()

def test2():
    from ProjUtils import DistDraw
    # point_seq = 5
    frame_seq = 5
    debug_device = R"VID_20240606_215815.mp4"
    cap = cv2.VideoCapture(debug_device)
    frame_count = 0
    process_count = 0
    ret, now_frame = cap.read()

    try:
        while True:
            last_frame = now_frame
            ret, now_frame = cap.read()
            # now_frame = cv2.cvtColor(now_frame, cv2.COLOR_BGR2GRAY)
            if not ret:
                break
            frame_count += 1
            if frame_count % frame_seq != 0:
                continue
            
            pts, dists = getPointDistances(last_frame, now_frame)
            img = drawDistPts(now_frame, pts, dists, 1)
            
            img = resizeImage(img, 1080)
            cv2.imshow('Image with Points and Distances', img)
            cv2.waitKey(0)
    except KeyboardInterrupt:
        print('Interrupted')
    finally:
        cv2.destroyAllWindows()
        cap.release()

init()
if __name__ == '__main__':
    test2()