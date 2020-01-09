import cv2
from utils import utils

video_path = "video/red_yellow/video5.mp4"
cap = cv2.VideoCapture(video_path)
frm = 0


while True:
    ret, frame = cap.read()
    frm += 1
    print(frm)
    if frm == 44:
        a = 1
    if ret:
        res = utils.field_detection(frame, "draw")
        cv2.imshow("result", res)
        cv2.waitKey(1)
    else:
        cv2.destroyAllWindows()
        break
