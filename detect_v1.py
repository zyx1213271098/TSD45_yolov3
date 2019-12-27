#coding:utf-8
#测试sample文件夹下的图像，并画框保存到result文件夹，2019.12.27
import argparse
import time
from sys import platform
import os
from models import *
from utils.datasets import *
from utils.utils import *
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def detect(
        cfg,
        data_cfg,
        weights,
        img_dir,
        output='output',  # output folder
        img_size=512,
        conf_thres=0.5,
        nms_thres=0.5,
        save_images=True,
        save_txt=False,
        webcam=False,
):
    #device = torch_utils.select_device()
    device = torch_utils.select_device(use_gpu=True)


    if os.path.exists(output):
        shutil.rmtree(output)  # delete output folder
    os.makedirs(output)  # make new output folder

    # Initialize model
    model = Darknet(cfg, img_size)

    # Load weights
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        _ = load_darknet_weights(model, weights)

    model.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        save_images = False
        dataloader = LoadWebcam(img_size=img_size)
    else:
        dataloader = LoadImages(img_dir, img_size=img_size)

    # Get classes and colors
    classes = load_classes(parse_data_cfg(data_cfg)['names'])
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]

    for i, (path, img, im0, vid_cap) in enumerate(dataloader):
        t = time.time()
        save_path = str(Path(output) / Path(path).name)

        # Get detections
        img = torch.from_numpy(img).unsqueeze(0).to(device)
        if ONNX_EXPORT:
            torch.onnx.export(model, img, 'weights/model.onnx', verbose=True)
            return
        pred = model(img)
        detections = non_max_suppression(pred, conf_thres, nms_thres)[0]

        if detections is not None and len(detections) > 0:
            # Rescale boxes from 416 to true image size
            scale_coords(img_size, detections[:, :4], im0.shape).round()

            # Print results to screen
            for c in detections[:, -1].unique():
                n = (detections[:, -1] == c).sum()
                print('%g %s' % (n, classes[int(c)]), end=', ')

            # Draw bounding boxes and labels of detections
            for *xyxy, conf, cls_conf, cls in detections:
                if save_txt:  # Write to file
                    with open(save_path + '.txt', 'a') as file:
                        file.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))

                # Add bbox to the image
                label = '%s %.2f' % (classes[int(cls)], conf)
                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])

        print('Done. (%.3fs)' % (time.time() - t))

        if webcam:  # Show live webcam
            cv2.imshow(weights, im0)

        if save_images:  # Save generated image with detections
            if dataloader.mode == 'video':
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (width, height))
                vid_writer.write(im0)

            else:
                cv2.imwrite(save_path, im0)

    if save_images and platform == 'darwin':  # macos
        os.system('open ' + output + ' ' + save_path)


if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    parser = argparse.ArgumentParser()
    # 配置文件
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-1cls.cfg', help='cfg file path')
    parser.add_argument('--data_cfg', type=str, default='tsd-data/cropImg_GT/tsd-1cls.data', help='tsd.data file path')
    parser.add_argument('--weights', type=str, default='result_dir_512_20epochs_1cls_Focal_loss/weights/best.pt', help='path to weights file')

    # 图片类
    parser.add_argument('--img_size', type=int, default=2048, help='size of each image dimension')
    parser.add_argument('--img_dir', type=str, default='./samples', help='dir of images')
    parser.add_argument('--outdir', type=str, default='./output', help='output dir')
    parser.add_argument('--save_img', type=bool, default=True, help='whether save result img')

    # NMS
    parser.add_argument('--conf_thres', type=float, default=0.1, help='object confidence threshold')
    parser.add_argument('--nms_thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
    parser.add_argument('--use_gpu', type=bool, default=True, help='use GPU or CPU')

    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect(
            opt.cfg,
            opt.data_cfg,
            opt.weights,
            opt.img_dir,
            img_size=opt.img_size,
            conf_thres=opt.conf_thres,
            nms_thres=opt.nms_thres,
            save_images=opt.save_img
        )