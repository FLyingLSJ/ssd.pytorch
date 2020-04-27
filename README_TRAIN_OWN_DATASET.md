 https://github.com/amdegroot/ssd.pytorch/issues/16 

### 配置开发环境

```bash
git clone https://github.com/FLyingLSJ/ssd.pytorch.git
```

- Python3+

```bash
pip install -r requirements.txt
```

- 训练可视化（安装后运行命令，在浏览器打开`http://localhost:8097` 即可进行可视化）

```bash
# First install Python server and client
pip install visdom
# Start the server (probably in a screen or tmux)
python -m visdom.server
```

### 准备数据集

构造与 VOC 相似结构的数据集，以下这些脚本可能对你有帮助

#### 对数据进行 VOC 数据结构的转换

```python
# wget https://static.leiphone.com/mask.zip
import os
import shutil
init_dir_list = ["Annotations", "JPEGImages", "ImageSets", "ImageSets/Main"]

src = "dataset/"
if not os.path.exists(src):
    os.mkdir(src)
for i in init_dir_list:
    path = os.path.join(src, i)
    if not os.path.exists(path):
        os.mkdir(path)

for path in ["train"]:
    file_list = ["./{}/".format(path) + i for i in os.listdir(path)]
    jpg_file = list(filter(lambda x:x.endswith("jpg"), file_list))
    xmk_file = list(filter(lambda x:x.endswith("xml"), file_list))
    
    print("In the {} phase, jpg file have {}, xml file have {}".format(path, len(jpg_file), len(xmk_file)))
    for i in jpg_file:
        dst = "dataset/JPEGImages/"
        shutil.copy(i, dst)    
        
    for i in xmk_file:    
        dst = "dataset/Annotations/"
        shutil.copy(i, dst)   
```



```python
import random
train_val = os.listdir("dataset/Annotations/")
train_val = list(map(lambda x:x.split(".")[0], train_val))
        
for i in train_val:
    with open("dataset/ImageSets/Main/trainval.txt", "a+") as f:
        f.write(i+"\n")       
```



#### 检查标注的名称是否准确/并统计直方图

```python
"""
该脚本用来检查标注的名称是否准确，并统计直方图
"""
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

# 改成你数据集的类别名    
CLASSES = ("face", "face_mask")

ann_path = "./dataset/Annotations/" # xml 文件所在的路径
xml_file_list = [ann_path+i for i in os.listdir(ann_path) if i.endswith("xml")] 

name_list = []
for xml_file in xml_file_list:
#     print(xml_file)
    xml_parse = ET.parse(xml_file).getroot()
    # [i for i in xml_parse]
    for obj in xml_parse.iter('object'):
        difficult = int(obj.find('difficult').text) == 1
        name = obj.find('name').text.lower().strip()
        name_list.append(name)
#         print(name)
#         bbox = obj.find('bndbox')
        if name not in CLASSES:
            print(name)
            print(xml_file)      

names = np.unique(name_list)
num_list = [np.sum(np.array(name_list) == i) for i in names]
plt.bar(names, num_list)
plt.show()
```

#### 检查图片的尺寸信息

```python
"""
统计标注文件 xml 文件中图片的尺寸信息，宽高和深度
"""
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


ann_path = "./dataset/Annotations/"  # xml 文件所在的路径 
xml_file_list = [ann_path+i for i in os.listdir(ann_path) if i.endswith("xml")] 

width_list = []
height_list= []
depth_list = []
for xml_file in xml_file_list:
        xml_parse = ET.parse(xml_file).getroot()
        # [i for i in xml_parse]
        for size in xml_parse.iter('size'):
            
            width = int(size.find("width").text)
            width_list.append(width)

            height = int(size.find("height").text)
            height_list.append(height)

            depth = int(size.find("depth").text)
            if depth == 1:
                print(xml_file)
            depth_list.append(depth)

print(np.unique(depth_list))  
plt.scatter(width_list, height_list)
plt.xlabel("width")
plt.ylabel("height")
```



#### 标注框的实际标注情况

```python
"""
查看标注框的实际标注情况
"""
import sys
import os
import cv2
import matplotlib.pyplot as plt

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

ann_path = "./dataset/Annotations/"
image_path = "./dataset/JPEGImages/"
xml_file_list = [ann_path+i for i in os.listdir(ann_path) if i.endswith("xml")] 

for xml_file in xml_file_list:
    xml_parse = ET.parse(xml_file).getroot()
    
    img_path = os.path.join(image_path, xml_file.split("/")[-1].split(".")[0]+".jpg")
    img = cv2.imread(img_path)
 
    for obj in xml_parse.iter('object'):
        bbox = obj.find('bndbox')
        pts = ['xmin', 'ymin', 'xmax', 'ymax']
        xmin = int(bbox.find("xmin").text) - 1
        ymin = int(bbox.find("ymin").text) - 1
        xmax = int(bbox.find("xmax").text) - 1
        ymax = int(bbox.find("ymax").text) - 1
#         print( (xmin, ymin), (xmax, ymax))
        img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), thickness=5)
    img = img[..., ::-1]
    plt.imshow(img)
    plt.show()
#     break

```

#### 检查标注是否准确

```python
import argparse
import sys
import cv2
import os

import os.path as osp
import numpy as np

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree  as ET


parser    = argparse.ArgumentParser(
            description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()

# 数据集所在的路径
parser.add_argument('--root', help='Dataset root directory path')

args = parser.parse_args()

CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

annopath = osp.join('%s', 'Annotations', '%s.{}'.format("xml"))
imgpath  = osp.join('%s', 'JPEGImages',  '%s.{}'.format("jpg"))

def vocChecker(image_id, width, height, keep_difficult = False):
    
    target   = ET.parse(annopath % image_id).getroot()
    res      = []

    for obj in target.iter('object'):

        difficult = int(obj.find('difficult').text) == 1

        if not keep_difficult and difficult:
            continue

        name = obj.find('name').text.lower().strip()
        bbox = obj.find('bndbox')

        pts    = ['xmin', 'ymin', 'xmax', 'ymax']
        bndbox = []

        for i, pt in enumerate(pts):

            cur_pt = int(bbox.find(pt).text) - 1
            # scale height or width
            cur_pt = float(cur_pt) / width if i % 2 == 0 else float(cur_pt) / height

            bndbox.append(cur_pt)

#         print(name)
        label_idx =  dict(zip(CLASSES, range(len(CLASSES))))[name]
        bndbox.append(label_idx)
        res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
        # img_id = target.find('filename').text[:-4]
#     print(res)
    try :
        print(np.array(res)[:,4])
        print(np.array(res)[:,:4])
    except IndexError:
        print("\nINDEX ERROR HERE !\n")
#        os.system("mv {} /root/code".format((annopath % image_id)))
        exit(0)
    return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]

if __name__ == '__main__' :
    
    i = 0
    path_list = []
    for name in sorted(os.listdir(osp.join(args.root,'Annotations'))):
    # as we have only one annotations file per image
        i += 1
        img    = cv2.imread(imgpath  % (args.root,name.split('.')[0]))
        height, width, channels = img.shape
        print("path : {}".format(annopath % (args.root,name.split('.')[0])))
        res = vocChecker((args.root, name.split('.')[0]), height, width)
    print("Total of annotations : {}".format(i))
```



### 下载权重文件

```bash
sh pretrained_model_download.sh
```

### 修改配置

修改 `ssd.pytorch/data/voc0712.py` 文件中的部分参数

- （约第 29行）变量 **VOC_ROOT** 将其修改为我们自己本地的 VOC 的路径
- （约第 98 行 ）修改 **image_sets** 的值为 **[('2007', 'trainval')]** ，因为我们只用 VOC2007 进行训练测试

修改 `/ssd.pytorch/data/config.py` 中的类别数，部分训练参数

### 训练

```bash
python train.py --cuda True --batch_size 32
```

### 测试

- 检查数据集是否准备好
- 检测权重文件 `weights/ssd_300_VOC0712.pth` 是否存在
- （可选）在 test.py 大概 108 行的位置：**testset = VOCDetection(args.voc_root, [('2007', 'test_mini')], None, VOCAnnotationTransform())** 修改 test_mini 成自己的文件，当然也可以不修改，我修改的原因是原始的 test.txt 里面有太多图片了，所以我自己建了一个较小的测试文件
- 运行以下代码，程序会自动对图片进行目标检测，检测后会在生成 eval 文件夹，并在下面生成一个 test1.txt 结果文件和图片

```bash
python test.py --cuda False 
```

- 如果你的电脑上有显卡的话则将 `--cuda` 设置成 True

### 检测

detect.py

### 问题解决

> RuntimeError: The shape of the mask [32, 8732] at index 0 does not match the shape of the indexed tensor [279424, 1] at index 0

- https://github.com/amdegroot/ssd.pytorch/issues/173 

> UserWarning: volatile was removed and now has no
> effect. Use `with torch.no_grad():` instead.

