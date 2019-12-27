#coding:utf-8
import json
import evaluate.anno_func as anno_func
import os

def eval3Cls(annosTargetPath,annosPredPath,minscore,type):

    annosTarget = json.loads(open(annosTargetPath).read())
    annosPred = json.loads(open(annosPredPath).read())

    sm = anno_func.eval_annos(annosTarget,annosPred,iou=0.5, types=type, minscore=minscore,check_type=False,minboxsize=0, maxboxsize=400)
    print(sm['report'])

    sm = anno_func.eval_annos(annosTarget, annosPred, iou=0.5, types=type, minscore=minscore, check_type=False,
                              minboxsize=0, maxboxsize=32)
    print(sm['report'])

    sm = anno_func.eval_annos(annosTarget, annosPred, iou=0.5, types=type, minscore=minscore, check_type=False,
                              minboxsize=32, maxboxsize=96)
    print(sm['report'])

    sm = anno_func.eval_annos(annosTarget, annosPred, iou=0.5, types=type, minscore=minscore, check_type=False,
                              minboxsize=96, maxboxsize=400)
    print(sm['report'])

    acc,rec = anno_func.get_acc_res(results_annos=annosPred,annos=annosTarget,type=type)
    MAP = anno_func.compute_AP(acc,rec)
    print("mAP:",MAP)


if __name__=='__main__':
    datadir =  '/home/zyx/data/TSD45_yolov3/tsd-data/test_json'
    #annosTargetPath = os.path.join(datadir,'type45_2048_target.json')
    annosTargetPath = os.path.join(datadir, 'annotations_src.json')
    annosPredPath = os.path.join(datadir,'type45_2048k_pred.json')
    #annosPredPath = os.path.join(datadir,'ours_result_annos.json')
    eval3Cls(annosTargetPath,annosPredPath,minscore=60,type=anno_func.type45)

