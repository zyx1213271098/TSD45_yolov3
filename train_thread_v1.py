import argparse

import torch.distributed as dist
from torch.utils.data import DataLoader

import test # Import test.py to get mAP after each epoch
from models import *
from utils.datasets import *
from utils.utils import *
from tensorboardX import SummaryWriter
import shutil
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import threading
from queue import Queue
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(0)

#定义数据队列data_queue，最多存储3个元素
data_queue = Queue(10)
train_done = False

def read_data(
        data_cfg,
        img_size=416,
        batch_size=16,
        num_workers=8,
        multi_scale=False
):

    if multi_scale:
        img_size = 512  # initiate with maximum multi_scale size
        num_workers = 8  # bug https://github.com/ultralytics/yolov3/issues/174
    else:
        torch.backends.cudnn.benchmark = True  # unsuitable for multiscale

    # Configure run
    train_path = parse_data_cfg(data_cfg)['train']

    # Classes
    cls = parse_data_cfg(data_cfg)['classes']

    start_epoch = 0

    # Dataset
    dataset = LoadImagesAndLabels(train_path, img_size=img_size, augment=True,cls=cls)

    sampler = None

    # Dataloader
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=True,
                            pin_memory=False,
                            collate_fn=dataset.collate_fn,
                            sampler=sampler)

    # Start training
    t = time.time()
    nB = len(dataloader)
    global train_done
    while not train_done:
        for i, (imgs, targets, img_path, _) in enumerate(dataloader):
            nT = len(targets)
            data_queue.put([imgs, targets, nT, nB, dataset.img_size])

            # Multi-Scale training (320 - 608 pixels) every 10 batches
            # if multi_scale and (i + 1) % 10 == 0:
            #     dataset.img_size = random.choice(range(10, 20)) * 32

            #print('data_queue size:',data_queue.qsize())
        event.clear()
        while not event.is_set() and train_done == False:
            time.sleep(0.5)
            pass

def train(
        cfg,
        data_cfg,
        batch_size=8,
        img_size=416,
        resume=False,
        epochs=270,
        accumulate=1,
        multi_scale=False,
        freeze_backbone=False,
        transfer=False,  # Transfer learning (train only YOLO layers)
):
    global train_done

    backbone_weights = 'weights' + os.sep  # backbone 权重路径

    result_dir = 'result_dir_512_20epochs_1cls'  # 结果保存路径
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    weights_dir = os.path.join(result_dir, 'weights')
    if not os.path.exists(weights_dir):
        os.mkdir(weights_dir)
    result_latest_weights = os.path.join(weights_dir, 'latest.pt')
    result_best_weights = os.path.join(weights_dir, 'best.pt')
    log_dir = os.path.join(result_dir, 'log_yolov3_512')
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.mkdir(log_dir)
    writer = SummaryWriter(comment='yolov3-512', log_dir=log_dir)

    if multi_scale:
        img_size = 512  # initiate with maximum multi_scale size
        num_workers = 0  # bug https://github.com/ultralytics/yolov3/issues/174
    else:
        torch.backends.cudnn.benchmark = True  # unsuitable for multiscale

    # cls
    cls = parse_data_cfg(data_cfg)['classes']

    # Initialize model
    model = Darknet(cfg, device,img_size).to(device)

    # Optimizer
    lr0 = 0.0001  # initial learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=lr0, momentum=0.9, weight_decay=0.0005)

    cutoff = -1  # backbone reaches to cutoff layer
    start_epoch = 0
    best_map = 0.1
    yl = get_yolo_layers(model)  # yolo layers
    nf = int(model.module_defs[yl[0] - 1]['filters'])  # yolo layer size (i.e. 255)

    if resume:  # Load previously saved model
        if transfer:  # Transfer learning
            chkpt = torch.load(backbone_weights + 'yolov3-spp.pt', map_location=device)
            model.load_state_dict({k: v for k, v in chkpt['model'].items() if v.numel() > 1 and v.shape[0] != 255},
                                  strict=False)
            for p in model.parameters():
                p.requires_grad = True if p.shape[0] == nf else False

        else:  # resume from latest.pt
            print("load latest weights...")
            chkpt = torch.load(result_latest_weights, map_location=device)  # load checkpoint
            model.load_state_dict(chkpt['model'])

        start_epoch = chkpt['epoch'] + 1
        if chkpt['optimizer'] is not None:
            optimizer.load_state_dict(chkpt['optimizer'])
            best_map = chkpt['best_map']
        del chkpt

    else:  # Initialize model with backbone (optional)
        if '-tiny.cfg' in cfg:
            cutoff = load_darknet_weights(model, backbone_weights + 'yolov3-tiny.conv.15')
        else:
            cutoff = load_darknet_weights(model, backbone_weights + 'darknet53.conv.74')

    # Set scheduler (reduce lr at epoch 10 15)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[10,15],gamma=0.1,last_epoch=start_epoch-1)

    # Start training
    t = time.time()
    model_info(model)

    for epoch in range(start_epoch,epochs):
        #torch.cuda.empty_cache()
        # event 事件开启，开始新一轮数据读取
        event.set()
        while data_queue.qsize()<10:
            print('--> data_queue size:',data_queue.qsize(),'  waiting for read data......')
            time.sleep(0.5)
            pass

        model.train()
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        print(('\n%8s%12s' + '%10s' * 7) % ('Epoch', 'Batch', 'xy', 'wh', 'conf', 'cls', 'total', 'nTargets', 'time'))

        # Updata scheduler
        scheduler.step()

        # Freeze backbone at epoch 0,unfreeze at epoch 1
        if freeze_backbone and epoch < 2:
            for name, p in model.named_parameters():
                if int(name.split('.')[1]) < cutoff:  # if layer < 75
                    p.requires_grad = False if epoch == 0 else True

        mloss = defaultdict(float)

        batch_i = 0
        while not data_queue.empty():
            imgs, targets, nT, nB,dataset_img_size = data_queue.get()
            imgs = imgs.to(device)
            targets = targets.to(device)
            #print('img_size:',img_size)
            if multi_scale and (batch_i) % 10 == 0:
                print('multi_scale img_size = %g' % dataset_img_size)

            # Plot images with bounding boxes
            plot_images = False
            if plot_images:
                fig = plt.figure(figsize=(10, 10))
                for ip in range(len(imgs)):
                    boxes = xywh2xyxy(targets[targets[:, 0] == ip, 2:6]).numpy().T * img_size
                    plt.subplot(4, 4, ip + 1).imshow(imgs[ip].numpy().transpose(1, 2, 0))
                    plt.plot(boxes[[0, 2, 2, 0, 0]], boxes[[1, 1, 3, 3, 1]], '.-')
                    plt.axis('off')
                fig.tight_layout()
                fig.savefig('batch_%g.jpg' % batch_i, dpi=fig.dpi)

            # SGD burn-in
            n_burnin = min(round(nB / 5 + 1), 1000)  # burn-in batches
            if epoch == 0 and batch_i <= n_burnin:
                lr = lr0 * (batch_i / n_burnin) ** 4
                for x in optimizer.param_groups:
                    x['lr'] = lr

            # Run model
            pred = model(imgs)

            # Build targets
            target_list = build_targets(model, targets)

            # Compute loss
            loss, loss_dict = compute_loss(pred, target_list, cls)

            # Compute gradient
            loss.backward()

            # Accumulate gradient for x batches before optimizing
            if (batch_i + 1) % accumulate == 0 or (batch_i + 1) == nB:
                optimizer.step()
                optimizer.zero_grad()

            # Running epoch-means of trached metrics
            for key, val in loss_dict.items():
                mloss[key] = (mloss[key] * batch_i + val) / (batch_i + 1)

            s = ('%8s%12s' + '%10.3g' * 7) % (
                '%g/%g' % (epoch, epochs - 1), '%g/%g' % (batch_i, nB - 1),
                mloss['xy'], mloss['wh'], mloss['conf'], mloss['cls'],
                mloss['total'], nT, time.time() - t)
            t = time.time()
            print(s)

            for key, value in mloss.items():
                writer.add_scalar('loss-' + key, value, (batch_i + 1 + epoch * nB))
            batch_i += 1
        # if epoch<10:
        #     continue

        #Calculate mAP
        with torch.no_grad():
            results = test.test(cfg, data_cfg, batch_size=batch_size, img_size=img_size, model=model,anno_file="tsd-data/cropImg_GT/json/cropped_test.json")
        writer.add_scalar('mAP',results[2],epoch)

        # Save training results
        save = True
        if save:
            # Save latest checkpoint
            chkpt = {'epoch': epoch,
                     'best_map': results[2],
                     'model': model.module.state_dict() if type(
                         model) is nn.parallel.DistributedDataParallel else model.state_dict(),
                     'optimizer': optimizer.state_dict()}
            torch.save(chkpt, result_latest_weights)

            # Update best map
            if results[2] > best_map:
                best_map = results[2]
                torch.save(chkpt, result_best_weights)

            # Save backup every 5 epochs (optional)
            backup_path = os.path.join(result_dir,'weights','backup%g.pt'%epoch)
            if epoch > 0 and epoch % 5 == 0:
                torch.save(chkpt, backup_path)

            del chkpt

        # Write epoch results
        result_txt_path = os.path.join(result_dir,'results.txt')
        with open(result_txt_path, 'a') as file:
            file.write(s + '%11.3g' * 3 % results + '\n')  # append P, R, mAP

    # 训练结束
    train_done = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='size of each image batch')
    parser.add_argument('--accumulate', type=int, default=1, help='accumulate gradient x batches before optimizing')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-1cls.cfg', help='cfg file path')
    parser.add_argument('--data-cfg', type=str, default='tsd-data/cropImg_GT/tsd-1cls.data', help='tsd.data file path')
    parser.add_argument('--multi-scale', default=True,action='store_true', help='random image sizes per batch 320 - 608')
    parser.add_argument('--img-size', type=int, default=512, help='pixels')
    parser.add_argument('--resume', default=False,action='store_true', help='resume training flag')
    parser.add_argument('--transfer', default=False,action='store_true', help='transfer learning flag')
    parser.add_argument('--num-workers', type=int, default=8, help='number of Pytorch DataLoader workers')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:9999', type=str, help='distributed training init method')
    parser.add_argument('--rank', default=0, type=int, help='distributed training node rank')
    parser.add_argument('--world-size', default=1, type=int, help='number of nodes for distributed training')
    parser.add_argument('--backend', default='nccl', type=str, help='distributed backend')
    opt = parser.parse_args()
    print(opt, end='\n\n')

    init_seeds()

    # 创建数据读取线程和train线程
    event = threading.Event()

    thread_data = threading.Thread(target=read_data, args=(opt.data_cfg,
                                                           opt.img_size,
                                                           opt.batch_size,
                                                           opt.num_workers,
                                                           opt.multi_scale))
    thread_train = threading.Thread(target=train, args=(opt.cfg,
                                                        opt.data_cfg,
                                                        opt.batch_size,
                                                        opt.img_size,
                                                        opt.resume or opt.transfer,
                                                        opt.epochs,
                                                        opt.accumulate,
                                                        opt.multi_scale,
                                                        opt.transfer))

    thread_data.start()
    time.sleep(0.5)
    thread_train.start()

    # 等待train()程序执行完毕
    thread_train.join()
    thread_data.join()
    print('Done.')