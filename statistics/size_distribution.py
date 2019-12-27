#coding:utf-8
import json
import os
import matplotlib.pyplot as plt

def SizeDistributin(json_path):
    annos = json.loads(open(json_path).read())
    sizes = [(round(obj['bbox']['ymax']-obj['bbox']['ymin'],3)) for k,img in annos['imgs'].items() for obj in img['objects']]
    plt.hist(x=sizes,bins=100)
    plt.savefig('hist.jpg')
    print('min size:',min(sizes),'max size:',max(sizes))

def ScoreDistributin(json_path):
    annos = json.loads(open(json_path).read())
    scores = [obj['score'] for k,img in annos['imgs'].items() for obj in img['objects']
              if (obj['bbox']['ymax']-obj['bbox']['ymin'])>96 and (obj['bbox']['ymax']-obj['bbox']['ymin'])<400]
    plt.hist(x=scores,bins=100)
    plt.savefig('score_hist_big.jpg')
    print('min size:', min(scores), 'max size:', max(scores))

if __name__=='__main__':
    #json_path = '/home/zyx/data/TSD45_yolov3/tsd-data/cropImg_GT/json/cropped_train.json'
    #SizeDistributin(json_path)
    json_path = '/home/zyx/data/TSD45_yolov3/tsd-data/test_json/type45_2048k_pred.json'
    ScoreDistributin(json_path)
