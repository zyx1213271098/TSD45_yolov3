import os
import torch.nn as nn

nn.Conv2d()

def imgPathTxt(Imgdir,type=None):
    datadir = os.path.join(Imgdir,type)
    #txtPath = os.path.join(Imgdir,type+'.txt')
    txtPath = os.path.join(Imgdir, 'test' + '.txt')
    with open(txtPath,'w') as f:
        for item in os.listdir(datadir):
            imgPath = os.path.join('/home/zyx/data/TSD45_yolov3/tsd-data/cropImg_GT/images/' + type,item)
            print(imgPath)
            f.write(imgPath+'\n')
        f.close()
if __name__=='__main__':
    Imgdir = '../tsd-data/cropImg_GT/images/'
    imgPathTxt(Imgdir,type='cropped_test')