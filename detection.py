import cv2
from utils import utils
from YOLOv3.detector import YOLOv3
from detect.team import TeamDetector
from detect.field import FieldDetector


video_path = "video/match1.mp4"
cap = cv2.VideoCapture(video_path)
frm = 0
yolo = YOLOv3("YOLOv3/cfg/yolo_v3.cfg", "YOLOv3/weights/yolov3.weights", "YOLOv3/cfg/coco.names", is_plot=False)


while True:
    ret, frame = cap.read()
    frm += 1
    if ret:
        # field = utils.field_detection(frame)
        # cv2.imshow("frame", field)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bbox, cls_confs, cls_ids = yolo(frame)
        #
        # FD = FieldDetector(frame, bbox)

        TC = TeamDetector(frame, bbox, cls_confs, cls_ids)
        res = TC.process()

        cv2.imshow("result", res)
        cv2.waitKey(1)
    else:
        cv2.destroyAllWindows()
        break


