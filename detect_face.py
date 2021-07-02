# -*- coding: UTF-8 -*-
import gi 
gi.require_version('Gtk','3.0')
import matplotlib
matplotlib.use('Agg')

import argparse
import time
from pathlib import Path
import os

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import copy

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression_face, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

def load_model(weights, device):
    model = attempt_load(weights, map_location=device)  # load FP32 model
    return model


def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6, 8]] -= pad[0]  # x padding
    coords[:, [1, 3, 5, 7, 9]] -= pad[1]  # y padding
    coords[:, :10] /= gain
    #clip_coords(coords, img0_shape)
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    coords[:, 4].clamp_(0, img0_shape[1])  # x3
    coords[:, 5].clamp_(0, img0_shape[0])  # y3
    coords[:, 6].clamp_(0, img0_shape[1])  # x4
    coords[:, 7].clamp_(0, img0_shape[0])  # y4
    coords[:, 8].clamp_(0, img0_shape[1])  # x5
    coords[:, 9].clamp_(0, img0_shape[0])  # y5
    return coords

def show_results(img, xywh, conf, landmarks, class_num, output_path=''):
    h,w,c = img.shape
    # print(h,w,c)
    tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
    x1 = int(xywh[0] * w - 0.5 * xywh[2] * w)
    y1 = int(xywh[1] * h - 0.5 * xywh[3] * h)
    x2 = int(xywh[0] * w + 0.5 * xywh[2] * w)
    y2 = int(xywh[1] * h + 0.5 * xywh[3] * h)
    cv2.rectangle(img, (x1,y1), (x2, y2), (0,255,0), thickness=tl, lineType=cv2.LINE_AA)

    # crop and output face
    crop_face = img[y1:y2,x1:x2]
    # print(y1,y2,x1,x2)
    if crop_face.size != 0 and output_path:
        output_file_name = output_path[:-4] + '_' + str(y1) + '_' + str(x1) + '.jpg'
    else:
        output_file_name = 'results/' + str(time.time()) + '.jpg'
    cv2.imwrite(output_file_name, crop_face)

    clors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255)]

    for i in range(5):
        point_x = int(landmarks[2 * i] * w)
        point_y = int(landmarks[2 * i + 1] * h)
        cv2.circle(img, (point_x, point_y), tl+1, clors[i], -1)

    tf = max(tl - 1, 1)  # font thickness
    label = str(conf)[:5]

    cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img



def detect_one(model, image_path, device, output_path):
    # Load model
    img_size = 2048
    conf_thres = 0.3
    iou_thres = 0.5

    orgimg = cv2.imread(image_path)  # BGR
    img0 = copy.deepcopy(orgimg)
    assert orgimg is not None, 'Image Not Found ' + image_path
    h0, w0 = orgimg.shape[:2]  # orig hw
    r = img_size / max(h0, w0)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1  else cv2.INTER_LINEAR
        img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

    imgsz = check_img_size(img_size, s=model.stride.max())  # check img_size

    img = letterbox(img0, new_shape=imgsz)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x416x416

    # Run inference
    t0 = time.time()

    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t1 = time_synchronized()
    pred = model(img)[0]

    # Apply NMS
    pred = non_max_suppression_face(pred, conf_thres, iou_thres)

    # print('img.shape: ', img.shape)
    # print('orgimg.shape: ', orgimg.shape)
    # file_predict = open(output_path[:-4] + '.txt', 'w+')


    # Process detections
    for i, det in enumerate(pred):  # detections per image
        gn = torch.tensor(orgimg.shape)[[1, 0, 1, 0]].to(device)  # normalization gain whwh
        gn_lks = torch.tensor(orgimg.shape)[[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]].to(device)  # normalization gain landmarks
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], orgimg.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class

            det[:, 5:15] = scale_coords_landmarks(img.shape[2:], det[:, 5:15], orgimg.shape).round()

            for j in range(det.size()[0]):
                xywh = (xyxy2xywh(det[j, :4].view(1, 4)) / gn).view(-1).tolist()
                conf = det[j, 4].cpu().numpy()
                landmarks = (det[j, 5:15].view(1, 10) / gn_lks).view(-1).tolist()
                class_num = det[j, 15].cpu().numpy()
                orgimg = show_results(orgimg, xywh, conf, landmarks, class_num, output_path)
                # file_predict.writelines(' '.join(map(str, xywh)))
                # file_predict.write('\n')

    cv2.imwrite(output_path[:-4] +'_result.jpg', orgimg)

from imutils.video import VideoStream
from imutils.video import FPS
import time

def detect_webcam(model, device):
    # Load model
    img_size = 2048
    conf_thres = 0.3
    iou_thres = 0.5

    # initialize the video stream and allow the camera sensor to warm up
    # print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    # vs = VideoStream(usePiCamera=True).start()

    time.sleep(1.0)
    # start the FPS counter
    fps = FPS().start()
    i = 0
    # loop over frames from the video file stream
    while True:
        # grab the frame from the threaded video stream and resize it
        # to 500px (to speedup processing)
        frame = vs.read()
        img0 = copy.deepcopy(frame)
        # frame = imutils.resize(frame, width=500)

        h0, w0 = frame.shape[:2]  # orig hw
        r = img_size / max(h0, w0)  # resize image to img_size

        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1  else cv2.INTER_LINEAR
            img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

        imgsz = check_img_size(img_size, s=model.stride.max())  # check img_size

        img = letterbox(img0, new_shape=imgsz)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x416x416

        # Run inference
        t0 = time.time()

        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img)[0]

        # Apply NMS
        pred = non_max_suppression_face(pred, conf_thres, iou_thres)

        # print('img.shape: ', img.shape)
        # print('orgimg.shape: ', orgimg.shape)
        # file_predict = open(output_path[:-4] + '.txt', 'w+')
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            gn = torch.tensor(frame.shape)[[1, 0, 1, 0]].to(device)  # normalization gain whwh
            gn_lks = torch.tensor(frame.shape)[[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]].to(device)  # normalization gain landmarks
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class

                det[:, 5:15] = scale_coords_landmarks(img.shape[2:], det[:, 5:15], frame.shape).round()

                for j in range(det.size()[0]):
                    xywh = (xyxy2xywh(det[j, :4].view(1, 4)) / gn).view(-1).tolist()
                    conf = det[j, 4].cpu().numpy()
                    landmarks = (det[j, 5:15].view(1, 10) / gn_lks).view(-1).tolist()
                    class_num = det[j, 15].cpu().numpy()
                    frame = show_results(frame, xywh, conf, landmarks, class_num, output_path)
                    # file_predict.writelines(' '.join(map(str, xywh)))
                    # file_predict.write('\n')
        
        # display the image to our screen
        cv2.imshow("Frame", frame)
        # print(time.time())
        i += 1
        key = cv2.waitKey(1000) & 0xFF
        # cv2.imwrite(output_path[:-4] +'_result.jpg', orgimg)
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
        # update the FPS counter
        fps.update()
    
    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    print(i)
    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='pretrained_models/yolov5s-face.pt', help='model.pt path(s)')
    parser.add_argument('--image', type=str, default='data/images/test.jpg', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='results', help='output folder to save crop faces')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    opt = parser.parse_args()
    # print(opt)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(opt.weights, device)
    if opt.image == '0':
        output_path = ''
        detect_webcam(model, device)
    else:
        output_path = os.path.join(opt.output, opt.image.split('/')[-1][:-4])
        detect_one(model, opt.image, device, output_path)




