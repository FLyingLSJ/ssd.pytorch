echo "Creating weights dir ..."
mkdir weights
cd weights

echo "Downloading vgg16_reducedfc.pth ..."
wget https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth

echo "Downloading ssd300_mAP_77.43_v2.pth ..."
wget https://s3.amazonaws.com/amdegroot-models/ssd300_mAP_77.43_v2.pth

echo "Downloading ssd_300_VOC0712.pth ..."
wget https://s3.amazonaws.com/amdegroot-models/ssd_300_VOC0712.pth