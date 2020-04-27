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


ann_path = "E:/VOC/VOC2007/Annotations/"  # xml 文件所在的路径 
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
plt.show()