import sys
import numpy as np
import cv2
import time
import torch

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression
from utils.torch_utils import select_device, time_synchronized

class Model:
    def __init__(self):
        # Useful
        self.weights = "./weights/best.pt"
        self.imgsz = 640

        # Useful but might not need to be in class attribute
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.classes = 0
        self.agnostic_nms = False

        # Initialize
        self.device = torch.device('cuda:0')
        print(self.device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check img_size
        if self.half:
            self.model.half()  # to FP16
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once

    def predict(self, image):
        # Resize
        img_org = letterbox(image, self.imgsz, self.stride)[0]

        try:
            # Convert
            img = img_org[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)

            # Run inference
            t0 = time.time()
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = self.model(img)[0]

            # Apply NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)[0]
            pred = pred[0].cpu().detach().numpy()
            #print(pred)

            t2 = time_synchronized()

            x1 = int(pred[0])
            y1 = int(pred[1])
            x2 = int(pred[2])
            y2 = int(pred[3])
            conf = round(pred[4], 2)

            cv2.rectangle(img_org, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_org, "conf : " + str(conf), (x1 + 1, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        except Exception as e:
            print("ERROR :", e)

        return img_org