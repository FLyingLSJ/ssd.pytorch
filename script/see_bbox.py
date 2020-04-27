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

ann_path = "E:/yanxishe/Safety_helmet/dataset/label_source/"
image_path = "E:/yanxishe/Safety_helmet/dataset/train/"
xml_file_list = [ann_path+i for i in os.listdir(ann_path) if i.endswith("xml")] 
xml_file_list = list(filter(lambda x:"1347.xml" in x, xml_file_list))
for xml_file in xml_file_list:
    print(xml_file)
    
    xml_parse = ET.parse(xml_file).getroot()
    
    img_path = os.path.join(image_path, xml_file.split("/")[-1].split("_")[-1].split(".")[0]+".jpg")
    print(img_path)
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
    """
    img = img[..., ::-1]
    plt.imshow(img)
    plt.show()
    """
    img = cv2.resize(img, (600, 600))
    cv2.imshow("img", img)
    cv2.waitKey(0)
cv2.destroyAllWindows()   
    
#     break
