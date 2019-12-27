import argparse
import json

from torch.utils.data import DataLoader
import evaluate.anno_func as anno_func
from models import *
from utils.datasets import *
from utils.utils import *
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def test(
        cfg,
        data_cfg,
        weights=None,
        batch_size=16,
        img_size=416,
        iou_thres=0.5,
        conf_thres=0.1,
        nms_thres=0.5,
        save_json=False,
        model=None,
        type='type45_512',
        anno_file='tsd-data/json/annotations_src.json',
):

    if model is None:
        #device = torch_utils.select_device()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(0)

        # Initialize model
        model = Darknet(cfg, img_size).to(device)

        # Load weights
        if weights.endswith('.pt'):  # pytorch format
            model.load_state_dict(torch.load(weights, map_location=device)['model'])
        else:  # darknet format
            _ = load_darknet_weights(model, weights)

        # if torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model)
    else:
        device = next(model.parameters()).device  # get model device

    # Configure run
    data_cfg = parse_data_cfg(data_cfg)
    test_path = data_cfg['valid']
    # if (os.sep + 'coco' + os.sep) in test_path:  # COCO dataset probable
    #     save_json = True  # use pycocotools

    # Classes
    cls = data_cfg['classes']

    # Dataloader
    dataset = LoadImagesAndLabels(test_path, img_size=img_size,cls=cls)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=4,
                            pin_memory=False,
                            collate_fn=dataset.collate_fn)

    model.eval()
    seen = 0
    print('%11s' * 5 % ('Image', 'Total', 'P', 'R', 'mAP'))
    mP, mR, mAP, mAPj = 0.0, 0.0, 0.0, 0.0
    jdict, tdict, stats, AP, AP_class = [], [], [], [], []
    coco91class = coco80_to_coco91_class()

    annos = json.loads(open(anno_file).read())
    annos_pred = {}
    annos_pred['imgs'] = {}
    annos_target = {}
    annos_target['imgs'] = {}
    classes = load_classes(data_cfg['names'])
    annos_type = anno_func.type45

    for batch_i, (imgs, targets, paths, shapes) in enumerate(tqdm(dataloader, desc='Calculating mAP')):
        targets = targets.to(device)
        imgs = imgs.to(device)

        #start_time_model = time.time()
        output = model(imgs)
        #end_time_model = time.time()
        #print('model_time:', end_time_model - start_time_model,end='    ')

        #start_time_nms = time.time()
        output = non_max_suppression(output, conf_thres=conf_thres, nms_thres=nms_thres)
        #end_time_nms = time.time()
        #print('nms_time:', end_time_nms - start_time_nms,end='    ')

        #start_time_correct = time.time()
        # Per image
        for si, pred in enumerate(output):
            labels = targets[targets[:, 0] == si, 1:]
            correct, detected = [], []
            tcls = torch.Tensor()
            seen += 1

            if pred is None:
                if len(labels):
                    tcls = labels[:, 0].cpu()  # target classes
                    stats.append((correct, torch.Tensor(), torch.Tensor(), tcls))
                continue

            #print('len(pred):', len(pred), end=' ')

            if True:
                imgid = Path(paths[si]).stem.split('/')[-1]
                annos_target['imgs'][imgid] = annos['imgs'][imgid]
                if('type3' in type):
                    annos_type = anno_func.type3
                    for c in range(len(annos['imgs'][imgid]['objects'])):
                        temp = annos['imgs'][imgid]['objects'][c]['category'][0]+'*'
                        annos_target['imgs'][imgid]['objects'][c]['category'] = annos['imgs'][imgid]['objects'][c]['category'][0]+'*'
                annos_pred['imgs'][imgid] = {}
                rpath = os.path.join('tsd-data/images/test', imgid + '.jpg')
                annos_pred['imgs'][imgid]['path'] = rpath
                annos_pred['imgs'][imgid]['objects'] = []
                boxes = pred[:, :4].clone()  # xyxy
                scale_coords(img_size, boxes, shapes[si])  # to original shape

                if boxes is None:
                    continue
                for di, d in enumerate(pred):
                    bbox={}
                    bbox['xmin'] = round(boxes[di][0].item(),6)
                    bbox['xmax'] = round(boxes[di][2].item(),6)
                    bbox['ymin'] = round(boxes[di][1].item(),6)
                    bbox['ymax'] = round(boxes[di][3].item(),6)

                    annos_pred['imgs'][imgid]['objects'].append(
                        {'score': 100 * float(d[4]), 'bbox': bbox, 'category': classes[int(d[6])]})

            if len(labels):
                # Extract target boxes as (x1, y1, x2, y2)
                tbox = xywh2xyxy(labels[:, 1:5]) * img_size  # target boxes
                tcls = labels[:, 0]  # target classes

                for *pbox, pconf, pcls_conf, pcls in pred:
                    if pcls not in tcls:
                        correct.append(0)
                        continue

                    # Best iou, index between pred and targets
                    iou, bi = bbox_iou(pbox, tbox).max(0)

                    # If iou > threshold and class is correct mark as correct
                    if iou > iou_thres and bi not in detected:# and pcls == tcls[bi]:
                        correct.append(1)
                        detected.append(bi)
                    else:
                        correct.append(0)
            else:
                # If no labels add number of detections as incorrect
                correct.extend([0] * len(pred))

            # Append Statistics (correct, conf, pcls, tcls)
            stats.append((correct, pred[:, 4].cpu(), pred[:, 6].cpu(), tcls.cpu()))
        #end_time_correct = time.time()
        #print('correct_time:', end_time_correct - start_time_correct)

    # Compute means
    stats_np = [np.concatenate(x, 0) for x in list(zip(*stats))]
    if len(stats_np):
        AP, AP_class, R, P = ap_per_class(*stats_np)
        mP, mR, mAP = P.mean(), R.mean(), AP.mean()
    # Print P, R, mAP
    print(('%11s%11s' + '%11.3g' * 3) % (seen, len(dataset), mP, mR, mAP))

    # Print mAP per class
    if len(stats_np):
        print('\nmAP Per Class:')
        names = load_classes(data_cfg['names'])
        for c, a in zip(AP_class, AP):
            print('%15s: %-.4f' % (names[c], a))
    minscore = 60


    sm = anno_func.eval_annos(annos_target, annos_pred, iou=0.5, types=annos_type, minscore=minscore,
                              check_type=True,
                              minboxsize=0, maxboxsize=32)
    print(sm['report'])
    sm = anno_func.eval_annos(annos_target, annos_pred, iou=0.5, types=annos_type, minscore=minscore,
                              check_type=True,
                              minboxsize=32, maxboxsize=96)
    print(sm['report'])
    sm = anno_func.eval_annos(annos_target, annos_pred, iou=0.5, types=annos_type, minscore=minscore,
                              check_type=True,
                              minboxsize=96, maxboxsize=400)
    print(sm['report'])

    targetdir = os.path.join('tsd-data/test_json/',type+'_target.json')
    with open(targetdir, 'w') as f:
        json.dump(annos_target, f)
        f.close()

    preddir = os.path.join('tsd-data/test_json/', type + '_pred.json')
    with open(preddir, 'w') as f:
        json.dump(annos_pred, f)
        f.close()

    # Return mAP
    return mP, mR, mAP


if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--batch-size', type=int, default=8, help='size of each image batch')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
    parser.add_argument('--data-cfg', type=str, default='tsd-data/tsd2048.data', help='coco.data file path')
    parser.add_argument('--weights', type=str, default='weights/backup/best.pt', help='path to weights file')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='iou threshold required to qualify as detected')
    parser.add_argument('--conf-thres', type=float, default=0.1, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--img-size', type=int, default=512, help='size of each image dimension')
    parser.add_argument('--anno-file',type=str, default='tsd-data/json/cropped_test.json' ,help='anno-file path')
    parser.add_argument('--test-type',type=str,default='type45_2048',help='test type')
    opt = parser.parse_args()

    TYPE = ['type45_512', 'type3_512', 'type45_2048','type3_2048']
    opt.test_type = TYPE[0]
    if opt.test_type == 'type45_512':
        opt.cfg = 'cfg/yolov3.cfg'
        opt.data_cfg = 'tsd-data/cropImg_GT/tsd.data'
        opt.weights = 'result_dir_512/weights/best.pt'
        opt.anno_file = 'tsd-data/cropImg_GT/json/cropped_test.json'

    elif opt.test_type == 'type3_512':
        opt.cfg = 'cfg/yolov3-3cls.cfg'
        opt.data_cfg = 'tsd-data/tsd-3cls.data'
        opt.weights = 'weights/backup/best-3cls.pt'
        opt.anno_file = 'tsd-data/json/cropped_test.json'

    elif opt.test_type == 'type45_2048':
        opt.cfg = 'cfg/yolov3.cfg'
        opt.data_cfg = 'tsd-data/cropImg_GT/tsd2048.data'
        opt.weights = 'weights/best-otherk.pt'
        opt.img_size = 2048
        opt.batch_size = 1
        opt.anno_file = 'tsd-data/cropImg_GT/json/annotations_src.json'


    print(opt, end='\n\n')

    with torch.no_grad():
        mAP = test(
            opt.cfg,
            opt.data_cfg,
            opt.weights,
            opt.batch_size,
            opt.img_size,
            opt.iou_thres,
            opt.conf_thres,
            opt.nms_thres,
            opt.save_json,
            type=opt.test_type,
            anno_file=opt.anno_file
        )