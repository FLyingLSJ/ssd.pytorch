import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import voc, coco
import os


class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train" 可选"train" 和 "test"
        size: input image size  输入网络的图片大小
        base: VGG16 layers for input, size of either 300 or 500 VGG16的网络层（修改fc后的）
        extras: extra layers that feed to multibox loc and conf layers 用于多尺度增加的网络
        head: "multibox head" consists of loc and conf conv layers 包含了各个分支的loc和conf
        num_classes: 类别数
    

    return:
        output: List, 返回loc, conf 和 候选框
    """

    def __init__(self, phase, size, base, extras, head, num_classes):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        # 配置config
        self.cfg = (coco, voc)[num_classes == 21] # 当 num_classes == 21 时 self.cfg=voc
        # 初始化先验框
        self.priorbox = PriorBox(self.cfg)
        with torch.no_grad():
            self.priors = self.priorbox.forward()
        self.size = size

        # SSD network
        # basebone 网络
        self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        # conv4_3后面的网络，L2 正则化
        """
        VGG 网络的 conv4_3 特征图大小 38x38，网络层靠前，norm 较大，
        需要加一个 L2 Normalization，以保证和后面的检测层差异不是很大。
        """
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)
        # 回归和分类网络
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list() # 用来保存要进行检测到特征图
        loc = list()
        conf = list()

        # apply vgg up to conv4_3 relu
        # vgg网络到conv4_3
        for k in range(23):
            x = self.vgg[k](x)
        # l2 正则化
        s = self.L2Norm(x)
        sources.append(s)

        # apply vgg up to fc7
        # conv4_3 到 fc
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)

        # apply extra layers and cache source layer outputs
        # extras 网络
        for k, v in enumerate(self.extras):
            # 把需要进行多尺度的网络输出存入 sources
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1: # 1, 3, 5, 7 时保存
                sources.append(x)

        # apply multibox head to source layers
        # 多尺度回归和分类网络
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous()) # permute 对张量的维度进行排列 https://pytorch.org/docs/stable/tensors.html?highlight=permute#torch.Tensor.permute
            conf.append(c(x).permute(0, 2, 3, 1).contiguous()) # contiguous() 返回张量本身  https://pytorch.org/docs/stable/tensors.html?highlight=contiguous#torch.Tensor.contiguous

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1) # 多尺度检测
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds 
                self.softmax(conf.view(conf.size(0), -1,
                             self.num_classes)),                # conf preds  
                self.priors.type(type(x.data))                  # default boxes 
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4), # loc的输出，size:(batch, 8732, 4)
                conf.view(conf.size(0), -1, self.num_classes), # conf的输出，size:(batch, 8732, 21)
                self.priors  # 生成所有的候选框 size([8732, 4])
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    """
    cfg: 配置信息
    """
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        # ceil_mode = True, 上采样使得 channel 75-->38
        elif v == 'C':            
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)] # https://pytorch.org/docs/stable/nn.html#maxpool2d
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm: # 是否有 BN 
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v # 更新下一层的输入
    # max_pooling (3,3,1,1)，对原始 VGG 网络修改 
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    # 新添加的网络层 1024x3x3 对原始 VGG 网络修改 
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    # 新添加的网络层 1024x1x1 对原始 VGG 网络修改 
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    # 结合到整体网络中
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    '''
    cfg：配置参数
    为后续多尺度提取，增加网络层
    '''
    layers = []
    # 初始输入通道为 1024，VGG最后一层输出是 1024
    in_channels = i
    #flag 用来选择 kernel_size= 1 or 3
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)] # 表达式 (1, 3)[flag] flag=True 时 为 3，否则为 1
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag # 反转flag
        in_channels = v # 更新 in_channels
    return layers


def multibox(vgg, extra_layers, cfg, num_classes):
    '''
    Args:
        vgg: 修改fc后的vgg网络
        extra_layers: 加在vgg后面的4层网络
        cfg: 网络参数，eg:[4, 6, 6, 6, 4, 4]
        num_classes: 类别，VOC为 20+背景=21
    Return:
        vgg, extra_layers
        loc_layers: 多尺度分支的回归网络
        conf_layers: 多尺度分支的分类网络
    '''
    loc_layers = []
    conf_layers = []
    vgg_source = [21, -2]
    # 第一部分，vgg 网络的 Conv2d-4_3(21层)， Conv2d-7_1(-2层，-1层是 relu 层)
    for k, v in enumerate(vgg_source):
        # 回归 box*4(坐标)
        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        # 置信度 box*(num_classes)                                    
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
    # 第二部分， cfg 从第三个（start=2）开始作为 box 的个数，而且用于多尺度提取的网络分别为 1,3,5,7 层
    for k, v in enumerate(extra_layers[1::2], start=2): # (k,v)->(2,1) (3,3) (4,5) (5,7)
        # 回归 box*4(坐标)
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]* 4, kernel_size=3, padding=1)]
        # 置信度 box*(num_classes)  
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]* num_classes, kernel_size=3, padding=1)]
    return vgg, extra_layers, (loc_layers, conf_layers)


base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [],
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [],
}
mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [],
}


def build_ssd(phase, size=300, num_classes=21):
    # 判断phase是否为满足的条件
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    # 判断size是否为满足的条件
    if size != 300:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300 (size=300) is supported!")
        return
    
    # 调用multibox，生成vgg,extras,head
    base_, extras_, head_ = multibox(vgg(base[str(size)], 3),
                                     add_extras(extras[str(size)], 1024),
                                     mbox[str(size)], 
                                     num_classes)
    return SSD(phase, size, base_, extras_, head_, num_classes)

    
