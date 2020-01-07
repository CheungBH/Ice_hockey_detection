import cv2
from utils import utils
import copy
from YOLOv3.detector import YOLOv3
import os
from detect.team import TeamDetector

num = 40
img_path = "frame/video3/video3_{}.jpg".format(num)
frame = cv2.imread(img_path)
os.makedirs("img/frame", exist_ok=True)
yolo = YOLOv3("YOLOv3/cfg/yolo_v3.cfg", "YOLOv3/weights/yolov3.weights", "YOLOv3/cfg/coco.names", is_plot=False)


field = utils.field_detection(frame)
cv2.imshow("frame", field)


origin_frame = copy.deepcopy(field)

frame = cv2.cvtColor(field, cv2.COLOR_BGR2RGB)
bbox, cls_confs, cls_ids = yolo(frame)

plotted_frame = utils.plot_bbox(frame, bbox, cls_confs, cls_ids)
plotted_frame = cv2.cvtColor(plotted_frame, cv2.COLOR_RGB2BGR)
cv2.imshow("bbox", plotted_frame)
cv2.imshow("frame", origin_frame)
#
# person_imgs = utils.get_person(origin_frame, bbox)
#
#
# TC = TeamDetector(origin_frame, bbox, cls_confs, cls_ids)
# res = TC.process()
# cv2.imshow("result", res)


cv2.waitKey(0)


