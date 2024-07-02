import TargetDist
from ProjUtils import DistDraw
import cv2

with open('apples.txt','w') as f:
    scale = 60/64.30488045122664
    for i in range(1,8): # 1..7
        path = f'apples/{i}.jpg'
        print(path)
        img = cv2.imread(path)
        ret = TargetDist.deal1frame(img)
        img = DistDraw.deal1frame(img, ret)
        # print(ret)
        xywh,cls,conf,dist1 = ret[0]
        dist2 = dist1 * scale
        ret = [(xywh,cls,conf,dist2)]
        # dist = ret[0][3] * scale
        dist0 = 60 * i
        # print(dist, dist0)
        print(dist2,dist1,dist0,sep=',',file=f)
        # cv2.imwrite(f'apples/out3_{i}.jpg', img)
        # cv2.imshow(path, img)
cv2.waitKey(0)
cv2.destroyAllWindows()