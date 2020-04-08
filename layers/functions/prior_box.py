from __future__ import division
from math import sqrt as sqrt
from itertools import product as product
import torch


class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.      
    1、计算先验框，根据feature map的每个像素生成box;
    2、框的中个数为： 38×38×4+19×19×6+10×10×6+5×5×6+3×3×4+1×1×4=8732
    3、 cfg: SSD的参数配置，字典类型   
    """
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.image_size = cfg['min_dim'] # 图片大小
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(cfg['aspect_ratios']) # 
        self.variance = cfg['variance'] or [0.1] # 若 cfg['variance'] 不含有 0.1 ，则 self.variance=cfg['variance']，否则就是 cfg['variance'] 和 0.1 的交集
        self.feature_maps = cfg['feature_maps'] # 多个尺度的特征图 [38, 19, 10, 5, 3, 1]
        self.min_sizes = cfg['min_sizes'] # [30, 60, 111, 162, 213, 264],
        self.max_sizes = cfg['max_sizes']
        self.steps = cfg['steps'] # [8, 16, 32, 64, 100, 300]
        self.aspect_ratios = cfg['aspect_ratios'] 
        self.clip = cfg['clip']
        self.version = cfg['name']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = [] # 用来存放 box的参数
        
        # 遍多尺度的 map: [38, 19, 10, 5, 3, 1]
        for k, f in enumerate(self.feature_maps):
            # 遍历每个像素
            for i, j in product(range(f), repeat=2):
                # k-th 层的feature map 大小
                f_k = self.image_size / self.steps[k] # 300/[8, 16, 32, 64, 100, 300] = [37.5, 18.75, 9.375, 4.6875, 3.0, 1.0]
                # unit center x,y
                # 每个框的中心坐标
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # aspect_ratio: 1
                # rel size: min_size
                # r==1, size = s_k， 正方形
                s_k = self.min_sizes[k]/self.image_size #   [0.1, 0.2, 0.37, 0.54, 0.71, 0.88]
                mean += [cx, cy, s_k, s_k]

                # aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                # r==1, size = sqrt(s_k * s_(k+1)), 正方形
                # (self.max_sizes[k]/self.image_size) [0.2, 0.37, 0.54, 0.71, 0.88, 1.05]
                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))  # [0.14142135623730953, 0.2720294101747089, 0.4469899327725402, 0.6191930232165088, 0.7904429138147802, 0.9612491872558333]
                mean += [cx, cy, s_k_prime, s_k_prime]

                # rest of aspect ratios
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]
                
        # back to torch land
        # 转化为 torch
        output = torch.Tensor(mean).view(-1, 4)
        # 归一化，把输出设置在 [0,1]
        if self.clip:
            output.clamp_(max=1, min=0)
        return output

        
        
# 调试代码
if __name__ == "__main__":
    # SSD300 CONFIGS
    voc = {
        'num_classes': 21,
        'lr_steps': (80000, 100000, 120000),
        'max_iter': 120000,
        'feature_maps': [38, 19, 10, 5, 3, 1],
        'min_dim': 300,
        'steps': [8, 16, 32, 64, 100, 300],
        'min_sizes': [30, 60, 111, 162, 213, 264],
        'max_sizes': [60, 111, 162, 213, 264, 315],
        'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
        'variance': [0.1, 0.2],
        'clip': True,
        'name': 'VOC',
    }
    box = PriorBox(voc)
    print('Priors box shape:', box.forward().shape)
    print('Priors box:\n',box.forward())   
