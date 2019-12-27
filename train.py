#coding:utf-8
# 使用git命令进行版本管理
# 2019.12.22
# 1.进行多线程预读取数据，只有单尺度
# 2.1cls的训练
# 3.不包含任何并行训练代码

import argparse
import time
import threading
from queue import Queue
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from utils.utils import *
from utils.datasets import *
from models import *
import test

# 定义数据队列data_queue，最多存储 num 个元素
num = 10;data_queue = Queue(num)
# 训练结束标志
train_done = False

def read_data(data_cfg,img_size,batch_size):
    '''
    功能：读取epoch数据暂存到全局队列data_queue中，读完一个epoch循环等待event信号，直到 train_done 标志为 True
    :param data_cfg: 数据配置文件路径
    :param img_size: 图片尺寸
    :param batch_size: dataloder读取数据的batch size
    :return: None
    '''
    # Configure run
    train_path = parse_data_cfg(data_cfg)['train']
    # Classes
    cls = parse_data_cfg(data_cfg)['classes']
    # Dataset
    dataset = LoadImagesAndLabels(train_path, img_size=img_size, augment=True,cls=cls)

    # Dataloader
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            pin_memory=False,
                            collate_fn=dataset.collate_fn,
                            sampler=None)

    # start read images and labels
    nB = len(dataloader)
    global train_done
    while not train_done:
        for i, (imgs, targets, img_path, _) in enumerate(dataloader):
            data_queue.put([imgs, targets, nB])
            #print('data_queue size:',data_queue.qsize())

        # 多线程通信事件clear,等待下个epoch的开始
        event.clear()
        while not event.is_set() and train_done == False:
            time.sleep(0.5)
            pass


def train(
        cfg,
        data_cfg,
        result_dir,
        use_gpu,
        batch_size=8,
        img_size=512,
        resume=False,
        epochs=20,
        accumulate=1,
        freeze_backbone=False,
        transfer=False,  # Transfer learning (train only YOLO layers)
):

    backbone_weights = 'weights' + os.sep  # backbone 权重路径

    # result 保存路径
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
    writer = SummaryWriter(comment='yolov3_512', log_dir=log_dir)
    # device
    device = torch_utils.select_device(use_gpu=use_gpu)
    # cls
    cls = parse_data_cfg(data_cfg)['classes']

    # Initialize model
    model = Darknet(cfg, device, img_size).to(device)

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
    global train_done
    model_info(model)
    for epoch in range(start_epoch,epochs):

        # 等待队列数据读满
        while data_queue.qsize()<num:
            print('--> data_queue size:',data_queue.qsize(),'  waiting for reading data...')
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
            imgs, targets, nB= data_queue.get();nT = len(targets)
            imgs = imgs.to(device)
            targets = targets.to(device)
            if (batch_i) % 10 == 0:   # 10个batch输出一次imgs.size()
                print('img_size = ',imgs.size())

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

        # event 事件开启，开始新一轮数据读取
        event.set()
        time.sleep(0.3)

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


if __name__=='__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    parser = argparse.ArgumentParser()
    # 模型和数据配置文件
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
    parser.add_argument('--data_cfg', type=str, default='tsd-data/cropImg_GT/tsd.data', help='tsd.data file path')
    parser.add_argument('--result_dir',type=str, default='result_dir_512_20epochs_K128',help='result dir name')
    parser.add_argument('--use_gpu', type=bool, default=True, help='whether use gpu')

    # 训练周期和batch size和多尺度等
    parser.add_argument('--epochs',type=int,default=20,help='number of epochs')
    parser.add_argument('--batch_size',type=int,default=8, help='size of each image batch')
    parser.add_argument('--img_size', type=int, default=512, help='pixels')
    parser.add_argument('--num_workers', type=int, default=8, help='number of Pytorch DataLoader workers')
    parser.add_argument('--accumulate', type=int, default=1, help='accumulate gradient x batches before optimizing')

    # 恢复训练
    parser.add_argument('--resume', default=True, action='store_true', help='resume training flag')
    parser.add_argument('--transfer', default=False, action='store_true', help='transfer learning flag')
    parser.add_argument('--backend', default='nccl', type=str, help='distributed backend')
    opt = parser.parse_args()
    print(opt, end='\n\n')

    # 初始化随机数种子
    init_seeds()

    # 创建数据读取线程和train线程
    event = threading.Event()

    thread_data = threading.Thread(target=read_data, args=(opt.data_cfg,
                                                           opt.img_size,
                                                           opt.batch_size))

    thread_train = threading.Thread(target=train, args=(opt.cfg,
                                                        opt.data_cfg,
                                                        opt.result_dir,
                                                        opt.use_gpu,
                                                        opt.batch_size,
                                                        opt.img_size,
                                                        opt.resume or opt.transfer,
                                                        opt.epochs,
                                                        opt.accumulate,
                                                        opt.transfer))

    thread_data.start()
    time.sleep(0.5)
    thread_train.start()

    # 等待train()程序执行完毕
    thread_train.join()
    thread_data.join()
    print('Done.')