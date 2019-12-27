import json
import pylab as pl
import random
import numpy as np
import cv2
import string
import os
# import anno_func

width_src = 2048.0
height_src = 2048.0
classes = ['io', 'po', 'wo', 'pl120', 'il80', 'pm30', 'i2', 'p12', 'pl100', 'w57', 'p26', 'ph5', 'pne', 'pl40', 'pl50',
           'pl20', 'pg', 'i5', 'p6', 'p3', 'pl80', 'pl60', 'pl70', 'p27', 'pm20', 'pl30', 'ph4', 'pm55', 'il60', 'p19',
           'pr40', 'i4', 'p23', 'pn', 'w32', 'pl5', 'p11', 'p5', 'ph4.5', 'ip', 'w13', 'w59', 'il100', 'p10', 'w55']
print(len(classes))

datadir = "/home/zyx/data/TT100k/data/annotations.json"

filedir = datadir
annos = json.loads(open(filedir).read())
annos['imgs'].keys()
for dict_key in annos['imgs'].keys():
    img_dict = annos['imgs'][dict_key]
    # print(img_dict)

    path = img_dict['path']  # 路径
    pic_id = img_dict['id']  # id
    imgobj_dict = img_dict['objects']

    train_flag = False
    # 判断 test or train
    if 'test' in path:
        file_path = '/home/zyx/data/TT100k/data/labels/test/'
    elif 'train' in path:
        file_path = '/home/zyx/data/TT100k/data/labels/train/'
    else:
        continue
    if not os.path.exists(file_path):
        os.mkdir(file_path)
    if pic_id==88586:
        print('88586')
    with open(file_path + str(pic_id) + '.txt', 'w') as file:
        for i in range(len(imgobj_dict)):

            category = imgobj_dict[i]['category']  # 类别
            xmax = imgobj_dict[i]['bbox']['xmax']
            xmin = imgobj_dict[i]['bbox']['xmin']
            ymax = imgobj_dict[i]['bbox']['ymax']
            ymin = imgobj_dict[i]['bbox']['ymin']

            if category not in classes:  # 其他类别 po io wo
                print(pic_id, file_path, category)
                continue

            if xmax < 0: xmax = 0
            if xmin < 0: xmin = 0
            if ymax < 0: ymax = 0
            if ymin < 0: ymin = 0

            x_center = round(((xmax - xmin) / 2.0 + xmin) / width_src, 6)
            y_center = round(((ymax - ymin) / 2.0 + ymin) / height_src, 6)
            width = round((xmax - xmin) / width_src, 6)
            height = round((ymax - ymin) / height_src, 6)

            info = str(classes.index(category)) + ' ' + str(x_center) + ' ' + str(y_center) + ' ' + str(
                width) + ' ' + str(height)
            file.write(info)
            info = ''
            file.write('\n')