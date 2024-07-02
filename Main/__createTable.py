# __createWidthTable.py
import json

output_filename = 'objlist.json'

# 原始数据
coco_default_labels = [
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light",
    # ... 其他标签 ...
]
default_widths = [
                  50, 0, 170, 0, 0,
                  0, 170, 0, 0, 0,
                  0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0,
                  0]
default_heights = [50, 0, 150, 0, 0,
                   0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0,
                   0]
known_labels = {0, 2}

if __name__ == '__main__':
    # 组织数据
    data = []
    for i, label in enumerate(coco_default_labels):
        if i in known_labels:
            data.append({
                'label': i,
                'label_name': label,
                'width': default_widths[i],
                'height': default_heights[i]
            })
        else:
            data.append({})
    
    # 存储为JSON文件
    with open(output_filename, 'w') as config_file:
        json.dump(data, config_file, indent=4)
    
    print('生成完成')
