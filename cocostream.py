
from pycocotools.coco import COCO
import requests

# instantiate COCO specifying the annotations json path
coco = COCO('...path_to_annotations/instances_train2017.json')
# Specify a list of category names of interest
catIds = coco.getCatIds(catNms=['person'])
# Get the corresponding image ids and images using loadImgs
imgIds = coco.getImgIds(catIds=catIds)
images = coco.loadImgs(imgIds)

for im in images:
    img_data = requests.get(im['coco_url']).content
    with open('...path_saved_ims/coco_person/' + im['file_name'], 'wb') as handler:
        handler.write(img_data)
        print(im)
