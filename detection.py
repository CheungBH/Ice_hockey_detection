import cv2
from utils import utils
import numpy as np
from YOLOv3.detector import YOLOv3

video_path = "video/match1.mp4"
cap = cv2.VideoCapture(video_path)
frm = 0
yolo = YOLOv3("YOLOv3/cfg/yolo_v3.cfg", "YOLOv3/weights/yolov3.weights", "YOLOv3/cfg/coco.names", is_plot=False)

while True:
    ret, frame = cap.read()
    frm += 1
    if ret:
        # field = utils.field_detection(frame)
        # cv2.imshow("frame", frame)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #res = yolo(conv_img)
        bbox, cls_conf, cls_ids = yolo(frame)
        print("There are {} bboxes in the image".format(len(bbox)))
        height, width = frame.shape[:2]
        #xmin ymin xmax ymax
        xmin = (bbox[:,0]-bbox[:,2]/2.0)*width
        ymin = (bbox[:,1]-bbox[:,3]/2.0)*height
        xmax = (bbox[:,0]+bbox[:,2]/2.0)*width
        ymax = (bbox[:,1]+bbox[:,3]/2.0)*height
        print(xmin)
        #cv2.namedWindow("yolo3", cv2.WINDOW_NORMAL)
        #cv2.resizeWindow("yolo3", 600,600)
        #cv2.imshow("yolo3",res[:,:,(2,1,0)])
        #cv2.waitKey(0)
        #boxes, cls_conf, cls_ids = yolo(frame)
        #print(boxes)


        
        for box in bbox:
            # get x1 x2 x3 x4
            x1 = int(round(((box[0] - box[2]/2.0) * width).item()))
            y1 = int(round(((box[1] - box[3]/2.0) * height).item()))
            x2 = int(round(((box[0] + box[2]/2.0) * width).item()))
            y2 = int(round(((box[1] + box[3]/2.0) * height).item()))
            #print(x1)
            # cls_conf = box[5]
            # cls_id = box[6]
            import random
            color = random.choices(range(256),k=3)
            color = [int(x) for x in np.random.randint(256, size=3)]
            # put texts and rectangles
            # if cls_id == 0:
            frame = cv2.putText(frame, "person", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        cv2.imshow("field", frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break 
        #cv2.waitKey(0)
    else:
        cv2.destroyAllWindows()
        break


