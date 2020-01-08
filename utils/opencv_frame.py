import cv2
import copy
import os
from utils import detect_light_dark


def process(img, wait=0):
    cv2.imshow("input", img)

    # detect_team(img, "red")
    # edge = cv2.Canny(img, 40, 300)
    # cv2.imshow("mask", edge)
    # cv2.moveWindow("mask", 400, 200)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("gray", gray)

    res, thresh = cv2.threshold(gray, 64, 255, cv2.THRESH_BINARY)
    cv2.imshow("thresh", thresh)
    cv2.moveWindow("input", 400, 500)
    cv2.moveWindow("thresh", 400, 200)

    ratio = detect_light_dark(thresh)

    # if ratio > 0.7:
    #     print("This player belongs to the white team\n")
    # else:
    #     print("This player belongs to the blue team\n")
    print(ratio)


    #
    # contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # len_ls = [len(con) for con in contours]
    # max_contour = contours[len_ls.index(max(len_ls))]
    #
    # contour = copy.deepcopy(img)
    #
    # x, y, w, h = cv2.boundingRect(max_contour)
    # cv2.rectangle(contour, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # cv2.imshow("contour", contour)

    cv2.waitKey(wait)


def process_img(img_path):
    process(cv2.imread(img_path))


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frm = 0
    while True:
        ret, frame = cap.read()
        frm += 1
        if ret:
            process(frame, wait=2)
        else:
            cv2.destroyAllWindows()
            break


def process_img_folder(folder_path):
    img_ls = [img_path for img_path in os.listdir(folder_path)]
    for img_name in img_ls:
        # print("Processing {}".format(img_name))
        process(cv2.imread(os.path.join(folder_path, img_name)), wait=500)


if __name__ == '__main__':
    path = "../img/frame/white"

    # path = "../video/video6.mp4"

    if path[-4:] == '.jpg' or  path[-4:] == '.png':
        process_img(path)
    elif path[-4:] == '.avi' or  path[-4:] == '.mp4':
        process_video(path)
    elif os.path.isdir(path):
        process_img_folder(path)
    else:
        raise ValueError("Wrong format!!")

