import cv2
from utils import utils
from YOLOv3.detector import YOLOv3


video_path = "video/video2.mp4"
cap = cv2.VideoCapture(video_path)
frm = 0
yolo = YOLOv3("YOLOv3/cfg/yolo_v3.cfg", "YOLOv3/weights/yolov3.weights", "YOLOv3/cfg/coco.names", is_plot=False)


while True:
    ret, frame = cap.read()
    frm += 1
    print(frm)
    if frm == 44:
        a = 1
    if ret:

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bbox, cls_confs, cls_ids = yolo(frame)

        res = utils.plot_bbox(frame, bbox, cls_confs, cls_ids)
        cv2.imshow("result", res)
        cv2.waitKey(1)
    else:
        cv2.destroyAllWindows()
        break
