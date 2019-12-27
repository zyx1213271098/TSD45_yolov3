import argparse
import time

import torch.distributed as dist
from torch.utils.data import DataLoader

import test  # Import test.py to get mAP after each epoch
from models import *
from utils.datasets import *
from utils.utils import *
from tensorboardX import SummaryWriter
import shutil
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def train(
        cfg,
        data_cfg,
        img_size=416,
        resume=False,
        epochs=270,
        batch_size=16,
        accumulate=1,
        multi_scale=False,
        freeze_backbone=False,
        num_workers=4,
        transfer=False  # Transfer learning (train only YOLO layers)
):
    weights = 'weights' + os.sep
    #latest = 'weights512/' + 'latest.pt'
    latest = 'weights512/' + 'backup10.pt'
    best = 'weights512/' + 'best.pt'
    #device = torch_utils.select_device()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(0)
    #device = torch.device("cpu")

    log_dir = './log_yolov3_512'
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.mkdir(log_dir)
    writer = SummaryWriter(comment='YOLOV3-TSD', log_dir=log_dir)

    if multi_scale:
        img_size = 608  # initiate with maximum multi_scale size
        num_workers = 0  # bug https://github.com/ultralytics/yolov3/issues/174
    else:
        torch.backends.cudnn.benchmark = True  # unsuitable for multiscale

    # Configure run
    train_path = parse_data_cfg(data_cfg)['train']

    # Classes
    cls = parse_data_cfg(data_cfg)['classes']

    # Initialize model
    model = Darknet(cfg, img_size).to(device)

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
            chkpt = torch.load(weights + 'yolov3-spp.pt', map_location=device)
            model.load_state_dict({k: v for k, v in chkpt['model'].items() if v.numel() > 1 and v.shape[0] != 255},
                                  strict=False)
            for p in model.parameters():
                p.requires_grad = True if p.shape[0] == nf else False

        else:  # resume from latest.pt
            print("load latest weights...")
            chkpt = torch.load(latest, map_location=device)  # load checkpoint
            model.load_state_dict(chkpt['model'])

        start_epoch = chkpt['epoch'] + 1
        if chkpt['optimizer'] is not None:
            optimizer.load_state_dict(chkpt['optimizer'])
            best_map = chkpt['best_map']
        del chkpt

    else:  # Initialize model with backbone (optional)
        if '-tiny.cfg' in cfg:
            cutoff = load_darknet_weights(model, weights + 'yolov3-tiny.conv.15')
        else:
            cutoff = load_darknet_weights(model, weights + 'darknet53.conv.74')

    # Set scheduler (reduce lr at epoch 250)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,15], gamma=0.1, last_epoch=start_epoch - 1)

    # Dataset
    dataset = LoadImagesAndLabels(train_path, img_size=img_size, augment=True,cls=cls)

    # Initialize distributed training  zyx
    # if torch.cuda.device_count() > 1:
    #     dist.init_process_group(backend=opt.backend, init_method=opt.dist_url, world_size=opt.world_size, rank=opt.rank)
    #     model = torch.nn.parallel.DistributedDataParallel(model)
    #     sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    # else:
    #     sampler = None

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
    model_info(model)
    nB = len(dataloader)
    n_burnin = min(round(nB / 5 + 1), 1000)  # burn-in batches
    for epoch in range(start_epoch, epochs):
        model.train()
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        print(('\n%8s%12s' + '%10s' * 7) % ('Epoch', 'Batch', 'xy', 'wh', 'conf', 'cls', 'total', 'nTargets', 'time'))

        # Update scheduler
        scheduler.step()

        # Freeze backbone at epoch 0, unfreeze at epoch 1
        if freeze_backbone and epoch < 2:
            for name, p in model.named_parameters():
                if int(name.split('.')[1]) < cutoff:  # if layer < 75
                    p.requires_grad = False if epoch == 0 else True

        mloss = defaultdict(float)  # mean loss
        for i, (imgs, targets, img_path, _) in enumerate(dataloader):
            print('img_shape:',imgs.size())
            imgs = imgs.to(device)
            targets = targets.to(device)

            nT = len(targets)

            if nT == 0:  # if no targets continue
                pass
                #continue

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
                fig.savefig('batch_%g.jpg' % i, dpi=fig.dpi)

            # SGD burn-in
            if epoch == 0 and i <= n_burnin:
                lr = lr0 * (i / n_burnin) ** 4
                for x in optimizer.param_groups:
                    x['lr'] = lr

            # Run model
            pred = model(imgs)

            # Build targets
            target_list = build_targets(model, targets)

            # Compute loss
            loss, loss_dict = compute_loss(pred, target_list)

            # Compute gradient
            loss.backward()

            # Accumulate gradient for x batches before optimizing
            if (i + 1) % accumulate == 0 or (i + 1) == nB:
                optimizer.step()
                optimizer.zero_grad()

            # Running epoch-means of tracked metrics
            for key, val in loss_dict.items():
                mloss[key] = (mloss[key] * i + val) / (i + 1)

            s = ('%8s%12s' + '%10.3g' * 7) % (
                '%g/%g' % (epoch, epochs - 1), '%g/%g' % (i, nB - 1),
                mloss['xy'], mloss['wh'], mloss['conf'], mloss['cls'],
                mloss['total'], nT, time.time() - t)
            t = time.time()
            print(s)

            for key,value in mloss.items():
                writer.add_scalar('loss-'+key, value, (i + 1 + epoch * len(dataloader)))

            #if mloss['total']==
            # Multi-Scale training (320 - 608 pixels) every 10 batches
            if multi_scale and (i + 1) % 10 == 0:
                dataset.img_size = random.choice(range(10, 20)) * 32
                print('multi_scale img_size = %g' % dataset.img_size)
            # if i>10:
            #     break
        # Calculate mAP
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
            torch.save(chkpt, latest)

            # Update best map
            if results[2] > best_map:
                best_map = results[2]
                torch.save(chkpt, best)

            # Save backup every 10 epochs (optional)
            if epoch > 0 and epoch % 10 == 0:
                torch.save(chkpt, 'weights512/' + 'backup%g.pt' % epoch)

            del chkpt

        # Write epoch results
        with open('results512.txt', 'a') as file:
            file.write(s + '%11.3g' * 3 % results + '\n')  # append P, R, mAP


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='size of each image batch')
    parser.add_argument('--accumulate', type=int, default=1, help='accumulate gradient x batches before optimizing')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
    parser.add_argument('--data-cfg', type=str, default='tsd-data/cropImg_GT/tsd.data', help='tsd.data file path')
    parser.add_argument('--multi-scale', default=True,action='store_true', help='random image sizes per batch 320 - 608')
    parser.add_argument('--img-size', type=int, default=512, help='pixels')
    parser.add_argument('--resume', default=True,action='store_true', help='resume training flag')
    parser.add_argument('--transfer', default=False,action='store_true', help='transfer learning flag')
    parser.add_argument('--num-workers', type=int, default=8, help='number of Pytorch DataLoader workers')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:9999', type=str, help='distributed training init method')
    parser.add_argument('--rank', default=0, type=int, help='distributed training node rank')
    parser.add_argument('--world-size', default=1, type=int, help='number of nodes for distributed training')
    parser.add_argument('--backend', default='nccl', type=str, help='distributed backend')
    opt = parser.parse_args()
    print(opt, end='\n\n')

    init_seeds()

    train(
        opt.cfg,
        opt.data_cfg,
        img_size=opt.img_size,
        resume=opt.resume or opt.transfer,
        transfer=opt.transfer,
        epochs=opt.epochs,
        batch_size=opt.batch_size,
        accumulate=opt.accumulate,
        multi_scale=opt.multi_scale,
        num_workers=opt.num_workers
    )
