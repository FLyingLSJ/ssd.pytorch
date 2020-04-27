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

已经办大家下载好了，数据集是 VOC2007，跑程序的话足够了。如果你想下载更多数据集，可以在 `ssd.pytorch/data/`下运行

```bash
# 下载 VOC2012数据集
sh data/scripts/VOC2007.sh
sh data/scripts/VOC2012.sh
# 下载 COCO 数据集，这个比较大
sh data/scripts/COCO2014.sh
```

### 下载权重文件

```bash
sh pretrained_model_download.sh
```

### 修改配置

修改 `ssd.pytorch/data/voc0712.py` 文件中的部分参数

- （约第 29行）变量 **VOC_ROOT** 将其修改为我们自己本地的 VOC 的路径
- （约第 98 行 ）修改 **image_sets** 的值为 **[('2007', 'trainval')]** ，因为我们只用 VOC2007 进行训练测试

### 训练

```bash
python train --cuda False 
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

### 问题解决

> RuntimeError: The shape of the mask [32, 8732] at index 0 does not match the shape of the indexed tensor [279424, 1] at index 0

- https://github.com/amdegroot/ssd.pytorch/issues/173 

> UserWarning: volatile was removed and now has no
> effect. Use `with torch.no_grad():` instead.

