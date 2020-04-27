from __future__ import print_function
import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
from data import VOC_ROOT, VOC_CLASSES as labelmap
from PIL import Image
from data import VOCAnnotationTransform, VOCDetection, BaseTransform, VOC_CLASSES
import torch.utils.data as data
from ssd import build_ssd
import matplotlib.pyplot as plt

import cv2
font = cv2.FONT_HERSHEY_SIMPLEX  # 指定字体

cuda = False
if torch.cuda.is_available():
    if cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')
    
# load net
model_path = "weights/ssd300_COCO_2000.pth"
num_classes = len(VOC_CLASSES) + 1 # +1 background    
net = build_ssd('test', 300, num_classes) # initialize SSD       
net.load_state_dict(torch.load(model_path))
    
net.eval()
if cuda:
    net = net.cuda()
    cudnn.benchmark = True
print('Finished loading model!')

transform = BaseTransform(net.size, (104, 117, 123))


base = "../test/"
img_list = [base + i for i in os.listdir(base)]

for img_path in img_list:
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
    x = Variable(x.unsqueeze(0))

    if cuda:
        x = x.cuda()

    y = net(x)      # forward pass
    detections = y.data
    # scale each detection back up to the image
    scale = torch.Tensor([img.shape[1], img.shape[0],
                         img.shape[1], img.shape[0]])
    pred_num = 0
    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= 0.5:
            """
            if pred_num == 0:
                with open(filename, mode='a') as f:
                    f.write('PREDICTIONS: '+'\n')
            """
            score = detections[0, i, j, 0]
            label_name = labelmap[i-1]
            pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
            coords = (pt[0], pt[1], pt[2], pt[3])
            """
            pred_num += 1            
            with open(filename, mode='a') as f:
                f.write(str(pred_num)+' label: '+label_name+' score: ' +
                        str(score) + ' '+' || '.join(str(c) for c in coords) + '\n')
            """
            j += 1
            img = cv2.rectangle(img, (pt[0], pt[1]), (pt[2], pt[3]), (0, 255, 0), 3)
            cv2.putText(img, label_name, (pt[0], pt[1]), font, 1, (255, 255, 255), 1, cv2.LINE_AA)  
            # 绘制的图像，文字，文字左下角的坐标,字体，字体颜色，厚度等
    img = img[..., ::-1]
    plt.imshow(img)
    plt.show()
    #cv2.imwrite(img_path.split("/")[-1], img)