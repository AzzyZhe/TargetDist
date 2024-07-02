# __createWidthTable.py
import json

output_filename = 'objlist.json'

names0 = { # model.names
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    4: 'airplane',
    5: 'bus',
    6: 'train',
    7: 'truck',
    8: 'boat',
    9: 'traffic light',
    10: 'fire hydrant',
    11: 'stop sign',
    12: 'parking meter',
    13: 'bench',
    14: 'bird',
    15: 'cat',
    16: 'dog',
    17: 'horse',
    18: 'sheep',
    19: 'cow',
    20: 'elephant',
    21: 'bear',
    22: 'zebra',
    23: 'giraffe',
    24: 'backpack',
    25: 'umbrella',
    26: 'handbag',
    27: 'tie',
    28: 'suitcase',
    29: 'frisbee',
    30: 'skis',
    31: 'snowboard',
    32: 'sports ball',
    33: 'kite',
    34: 'baseball bat',
    35: 'baseball glove',
    36: 'skateboard',
    37: 'surfboard',
    38: 'tennis racket',
    39: 'bottle',
    40: 'wine glass',
    41: 'cup',
    42: 'fork',
    43: 'knife',
    44: 'spoon',
    45: 'bowl',
    46: 'banana',
    47: 'apple',
    48: 'sandwich',
    49: 'orange',
    50: 'broccoli',
    51: 'carrot',
    52: 'hot dog',
    53: 'pizza',
    54: 'donut',
    55: 'cake',
    56: 'chair',
    57: 'couch',
    58: 'potted plant',
    59: 'bed',
    60: 'dining table',
    61: 'toilet',
    62: 'tv',
    63: 'laptop',
    64: 'mouse',
    65: 'remote',
    66: 'keyboard',
    67: 'cell phone',
    68: 'microwave',
    69: 'oven',
    70: 'toaster',
    71: 'sink',
    72: 'refrigerator',
    73: 'book',
    74: 'clock',
    75: 'vase',
    76: 'scissors',
    77: 'teddy bear',
    78: 'hair drier',
    79: 'toothbrush'
}

# 原始数据
data0 = {
    0: ['person', (50, 170, -1)],
    1: ['bicycle', (150, 80, -1)],
    2: ['car', (180, 150, -1)],
    3: ['motorcycle', (160, 90, -1)],
    4: ['airplane', (-1, -1, -1)],
    5: ['bus', (250, 300, -1)],
    6: ['train', (-1, -1, -1)],
    7: ['truck', (-1, -1, -1)],
    8: ['boat', (-1, -1, -1)],
    9: ['traffic light', (-1, -1, -1)],
    10: ['fire hydrant', (-1, -1, -1)],
    11: ['stop sign', (-1, -1, -1)],
    12: ['parking meter', (-1, -1, -1)],
    13: ['bench', (-1, -1, -1)],
    14: ['bird', (30, 30, 30)],
    15: ['cat', (-1, -1, -1)],
    16: ['dog', (-1, -1, -1)],
    17: ['horse', (-1, -1, -1)],
    18: ['sheep', (-1, -1, -1)],
    19: ['cow', (-1, -1, -1)],
    20: ['elephant', (-1, -1, -1)],
    21: ['bear', (-1, -1, -1)],
    22: ['zebra', (-1, -1, -1)],
    23: ['giraffe', (-1, -1, -1)],
    24: ['backpack', (-1, -1, -1)],
    25: ['umbrella', (-1, -1, -1)],
    26: ['handbag', (-1, -1, -1)],
    27: ['tie', (-1, -1, -1)],
    28: ['suitcase', (-1, -1, -1)],
    29: ['frisbee', (-1, -1, -1)],
    30: ['skis', (-1, -1, -1)],
    31: ['snowboard', (-1, -1, -1)],
    32: ['sports ball', (-1, -1, -1)],
    33: ['kite', (-1, -1, -1)],
    34: ['baseball bat', (-1, -1, -1)],
    35: ['baseball glove', (-1, -1, -1)],
    36: ['skateboard', (-1, -1, -1)],
    37: ['surfboard', (-1, -1, -1)],
    38: ['tennis racket', (-1, -1, -1)],
    39: ['bottle', (10, 25, -1)],
    40: ['wine glass', (-1, -1, -1)],
    41: ['cup', (12, 12, -1)],
    42: ['fork', (-1, -1, -1)],
    43: ['knife', (-1, -1, -1)],
    44: ['spoon', (-1, -1, -1)],
    45: ['bowl', (-1, -1, -1)],
    46: ['banana', (30, 10, -1)],
    47: ['apple', (9, 7, 9)],
    48: ['sandwich', (-1, -1, -1)],
    49: ['orange', (10, 10, 10)],
    50: ['broccoli', (-1, -1, -1)],
    51: ['carrot', (-1, -1, -1)],
    52: ['hot dog', (-1, -1, -1)],
    53: ['pizza', (-1, -1, -1)],
    54: ['donut', (-1, -1, -1)],
    55: ['cake', (-1, -1, -1)],
    56: ['chair', (40, -1, -1)],
    57: ['couch', (-1, -1, -1)],
    58: ['potted plant', (-1, -1, -1)],
    59: ['bed', (-1, -1, -1)],
    60: ['dining table', (-1, -1, -1)],
    61: ['toilet', (-1, -1, -1)],
    62: ['tv', (-1, -1, -1)],
    63: ['laptop', (-1, -1, -1)],
    64: ['mouse', (-1, -1, -1)],
    65: ['remote', (-1, -1, -1)],
    66: ['keyboard', (-1, -1, -1)],
    67: ['cell phone', (-1, -1, -1)],
    68: ['microwave', (-1, -1, -1)],
    69: ['oven', (-1, -1, -1)],
    70: ['toaster', (-1, -1, -1)],
    71: ['sink', (-1, -1, -1)],
    72: ['refrigerator', (-1, -1, -1)],
    73: ['book', (25, 35, -1)],
    74: ['clock', (-1, -1, -1)],
    75: ['vase', (10, 25, -1)],
    76: ['scissors', (-1, -1, -1)],
    77: ['teddy bear', (-1, -1, -1)],
    78: ['hair drier', (-1, -1, -1)],
    79: ['toothbrush', (-1, -1, -1)]
}


if __name__ == '__main__':
    # 组织数据
    data = []
    for i in range(len(data0)):
        label, whl = data0[i]
        if whl != (-1, -1, -1):
            w, h, l = whl
            data.append({
                'label': i,
                'label_name': label,
                'width': w,
                'height': h
            })
        else:
            data.append({})
    
    # 存储为JSON文件
    with open(output_filename, 'w') as config_file:
        json.dump(data, config_file, indent=4)
    
    print('生成完成')
