import cv2
from utils import utils
from YOLOv3.detector import YOLOv3


img_path = "frame/video1/video1_80.jpg"
cap = cv2.VideoCapture(img_path)
yolo = YOLOv3("YOLOv3/cfg/yolo_v3.cfg", "YOLOv3/weights/yolov3.weights", "YOLOv3/cfg/coco.names", is_plot=False)
save = True


img = cv2.imread(img_path)

frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
bbox, cls_confs, cls_ids = yolo(frame)

if save:
    for

res = utils.plot_bbox(frame, bbox, cls_confs, cls_ids)
cv2.imshow("result", res)
cv2.waitKey(0)

