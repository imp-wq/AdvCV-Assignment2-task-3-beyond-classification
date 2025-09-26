mkdir -p data/coco
cd data/coco
# 图像
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip
# 标注
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip
