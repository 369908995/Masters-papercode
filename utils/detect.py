import torch
from numpy import random
from models.experimental import attempt_load
from utils.datasets import  MyLoadImages,LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, \
    scale_coords, set_logging
# from utils.plots import plot_one_box
# from utils.torch_utils import select_device, load_classifier
 
class simulation_opt:
    def __init__(self, weights='weights/yolov7.pt',
                 img_size = 640, conf_thres = 0.95,
                 iou_thres = 0.95,device='', view_img= False,
                 classes = None, agnostic_nms = False,
                 augment = False, update = False, exist_ok = False):
        self.weights = weights
        self.source = None
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = device
        self.view_img = view_img
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.augment =augment
        self.update = update
        self.exist_ok = exist_ok
 
class detectapi_yolov7:
    def __init__(self, weights, img_size=640):
        self.opt = simulation_opt(weights=weights, img_size=img_size)
        weights, imgsz = self.opt.weights, self.opt.img_size
 
        # Initialize
        set_logging()
        self.device = select_device(self.opt.device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
 
        # Load model
        self.model = attempt_load(weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(imgsz, s=self.stride)  # check img_size
 
        if self.half:
            self.model.half()  # to FP16
 
        # Second-stage classifier
        self.classify = False
        if self.classify:
            self.modelc = load_classifier(name='resnet101', n=2)  # initialize
            self.modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=self.device)['model']).to(self.device).eval()
 
        # read names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]
 
    def detect(self, source):  # 使用时，调用这个函数
        if type(source) != list:
            raise TypeError('source must be a list which contain  pictures read by cv2')
        dataset = MyLoadImages(source, img_size=self.imgsz, stride=self.stride)#imgsz
        # dataset = LoadImages(source, img_size=self.imgsz, stride=self.stride)#
        # 原来是通过路径加载数据集的，现在source里面就是加载好的图片，所以数据集对象的实现要
        # 重写。修改代码后附。在utils.dataset.py上修改。
        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once
        #t0 = time.time()
        result = []
        '''
        for path, img, im0s, vid_cap in dataset:'''
        for img, im0s in dataset:
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
 
            # Inference
            #t1 = time_synchronized()
            pred = self.model(img, augment=self.opt.augment)[0]
 
            # Apply NMS
            pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes, agnostic=self.opt.agnostic_nms)
            #t2 = time_synchronized()
 
            # Apply Classifier
            if self.classify:
                pred = apply_classifier(pred, self.modelc, img, im0s)
                # Print time (inference + NMS)
                #print(f'{s}Done. ({t2 - t1:.3f}s)')
                # Process detections
            det = pred[0]  # 原来的情况是要保持图片，因此多了很多关于保持路径上的处理。另外，pred
            # 其实是个列表。元素个数为batch_size。由于对于我这个api，每次只处理一个图片，
            # 所以pred中只有一个元素，直接取出来就行，不用for循环。
            im0 = im0s.copy()  # 这是原图片，与被传进来的图片是同地址的，需要copy一个副本，否则，原来的图片会受到影响
            # s += '%gx%g ' % img.shape[2:]  # print string
            # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            result_txt = []
            # 对于一张图片，可能有多个可被检测的目标。所以结果标签也可能有多个。
            # 每被检测出一个物体，result_txt的长度就加一。result_txt中的每个元素是个列表，记录着
            # 被检测物的类别引索，在图片上的位置，以及置信度
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # Write results
 
                for *xyxy, conf, cls in reversed(det):
                    # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (int(cls.item()), [int(_.item()) for _ in xyxy], conf.item())  # label format
                    result_txt.append(line)
                    label = f'{self.names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=5)
            result.append((im0, result_txt))  # 对于每张图片，返回画完框的图片，以及该图片的标签列表。
        return result, self.names
class slid_window:
    def __init__(self,wight,cls):
        self.wight=wight
        self.windows=[0 for i in range(wight)]
        self.N=0
        self.S='empty'
        self.T_full=0
        self.T_middle=0
        self.cls=cls
        self.police=0
    def slid(self,x1,y1,x2,y2,conf):
        if(conf==None):
            # print('no no no no')
            self.windows.append(0)
            flag=self.windows.pop(0)
        else:
            self.windows.append(1)
            self.N+=1
            flag=self.windows.pop(0)
        self.N=self.N-1 if flag else self.N

        if self.N==self.wight:
            S='full'
        elif self.N==0:
            S='empty'
        else:
            S='middle'
        
        if S=='empty':
            if self.S=='full':
                self.S='empty'
                self.T_full=0
                self.police=6
            elif self.S=='empty':
                self.police=0
            else:
                self.S='empty'
                self.T_middle=0
                self.police=3
        elif S=='middle':
            if self.S=='full':
                self.S='middle'
                self.T_full=0
                self.T_middle+=1
                self.police=7
            elif self.S=='empty':
                self.S='middle'
                self.T_middle+=1
                self.police=1
            else:
                self.T_middle+=1
                self.police=4
        else:
            if self.S=='full':
                self.T_full+=1
                self.police=8
            elif self.S=='empty':
                self.S='full'
                self.T_full+=1
                self.police=2
            else:
                self.S='full'
                self.T_full+=1
                self.T_middle=0
                self.police=5
        return self.S,self.police,self.T_full,self.T_middle

# YOLOv5  by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python path/to/detect.py --weights yolov5s.pt --source 0              # webcam
                                                             img.jpg        # image
                                                             vid.mp4        # video
                                                             path/          # directory
                                                             path/*.jpg     # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python path/to/detect.py --weights yolov5s.pt                 # PyTorch
                                         yolov5s.torchscript        # TorchScript
                                         yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                         yolov5s.xml                # OpenVINO
                                         yolov5s.engine             # TensorRT
                                         yolov5s.mlmodel            # CoreML (MacOS-only)
                                         yolov5s_saved_model        # TensorFlow SavedModel
                                         yolov5s.pb                 # TensorFlow GraphDef
                                         yolov5s.tflite             # TensorFlow Lite
                                         yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

import ssl

ssl._create_default_https_context = ssl._create_unverified_context



FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync



@torch.no_grad()
class detectapi_yolov5:
    def __init__(self,weight,
                data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
                imgsz=[640, 640],  # inference size (height, width)
                conf_thres=0.85,  # confidence threshold
                iou_thres=0.85,  # NMS IOU threshold
                max_det=1000,  # maximum detections per image
                device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                save_img=False,
                view_img=False,  # show results
                save_txt=False,  # save results to *.txt
                save_conf=False,  # save confidences in --save-txt labels
                save_crop=False,  # save cropped prediction boxes
                nosave=False,  # do not save images/videos
                classes=None,  # filter by class: --class 0, or --class 0 2 3
                agnostic_nms=True,  # class-agnostic NMS
                augment=False,  # augmented
                
                
                # inference
                visualize=False,  # visualize features
                update=False,  # update all models
                project=ROOT / 'runs/detect',  # save results to project/name
                name='exp',  # save results to project/name
                exist_ok=False,  # existing project/name ok, do not increment
                line_thickness=3,  # bounding box thickness (pixels)
                hide_labels=False,  # hide labels
                hide_conf=False,  # hide confidences
                half=False,  # use FP16 half-precision inference
                dnn=False,  # use OpenCV DNN for ONNX inference

    ) :
        self.weights=weight
        self.conf_thres=conf_thres
        self.iou_thres=iou_thres
        self.max_det=max_det
        self.save_img=save_img
        self.view_img=view_img
        self.save_txt=save_txt
        self.save_conf=save_conf
        self.save_crop=save_crop
        self.nosave=nosave
        self.classes=classes
        self.agnostic_nms=agnostic_nms
        self.augment=augment
        self.visualize=visualize        
        self.update=update
        self.project=project
        self.name=name
        self.exist_ok=exist_ok
        self.line_thickness=line_thickness
        self.hide_labels=hide_labels
        self.hide_conf=hide_conf
        self.half=half
        self.dnn=dnn
        self.data=data
         # Load model
        self.device = select_device(device)
        self.model = DetectMultiBackend(self.weights, device=self.device, dnn=self.dnn, data=self.data)
        self.stride, self.names, self.pt, self.jit, self.onnx, self.engine = self.model.stride, self.model.names, self.model.pt, self.model.jit, self.model.onnx, self.model.engine
        self.imgsz = check_img_size(imgsz, s=self.stride)  # check image size

        # Half
        self.half = (self.pt or self.jit or self.engine) and self.device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
        if self.pt or self.jit:
            self.model.model.half() if self.half else self.model.model.float()
    def run(self,source):

        result = []
        weights=self.weights

        # Dataloader
        if type(source) != list:
            raise TypeError('source must be a list which contain  pictures read by cv2')
        dataset = MyLoadImages(source, img_size=self.imgsz, stride=self.stride)#imgsz
       
        # Run inference
        self.model.warmup(imgsz=(1, 3, *self.imgsz), half=self.half)  # warmup
        # dt, seen = [0.0, 0.0, 0.0], 0
        half=self.half
        for img, im0s in dataset:
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = self.model(img, augment=self.augment, visualize=self.visualize)

            # NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)

######################################################################################################################################
            det = pred[0]  # 原来的情况是要保持图片，因此多了很多关于保持路径上的处理。另外，pred
            # 其实是个列表。元素个数为batch_size。由于对于我这个api，每次只处理一个图片，
            # 所以pred中只有一个元素，直接取出来就行，不用for循环。
            im0 = im0s.copy()  # 这是原图片，与被传进来的图片是同地址的，需要copy一个副本，否则，原来的图片会受到影响
            # s += '%gx%g ' % img.shape[2:]  # print string
            # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            result_txt = []
            # 对于一张图片，可能有多个可被检测的目标。所以结果标签也可能有多个。
            # 每被检测出一个物体，result_txt的长度就加一。result_txt中的每个元素是个列表，记录着
            # 被检测物的类别引索，在图片上的位置，以及置信度
            annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # Write results
                for *xyxy, conf, cls in reversed(det):
                        line = (int(cls.item()), [int(_.item()) for _ in xyxy], conf.item())
                        result_txt.append(line)
                        c = int(cls)  # integer class
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        result.append((im0, result_txt))
            return result,self.names