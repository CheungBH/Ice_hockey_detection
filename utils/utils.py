import cv2
import numpy as np


def field_detection(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res, thresh1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    len_ls = [len(con) for con in contours]
    max_contour = contours[len_ls.index(max(len_ls))]
    x, y, w, h = cv2.boundingRect(max_contour)
    height, width = img.shape[0], img.shape[1]
    return cut_image(img, left=x, right=width-w, top=y, bottom=height-h)


def cut_image(img, bottom=0, top=0, left=0, right=0):
    height, width = img.shape[0], img.shape[1]
    return np.asarray(img[top: height - bottom, left: width - right])


def count_nonblack_np(img):
    """Return the number of pixels in img that are not black.
    img must be a Numpy array with colour values along the last axis.

    """
    return img.any(axis=-1).sum()


# def detect_team(image, color, show=True):
#     # define the list of boundaries
#     if color == "yellow":
#         boundaries = [
#             # ([17, 15, 100], [50, 56, 200]),  # red
#             ([25, 146, 190], [96, 174, 250])  # yellow
#         ]
#     elif color == "red":
#         boundaries = [
#             ([17, 15, 100], [50, 56, 200]),  # red
#             # ([25, 146, 190], [96, 174, 250])  # yellow
#         ]
#     elif color == "black":
#         boundaries = [
#             ([0, 0, 0], [180, 255, 46]),  # red
#             # ([25, 146, 190], [96, 174, 250])  # yellow
#         ]
#     else:
#         raise ValueError("Wrong color")
#
#     i = 0
#     for (lower, upper) in boundaries:
#         # create NumPy arrays from the boundaries
#         lower = np.array(lower, dtype="uint8")
#         upper = np.array(upper, dtype="uint8")
#
#         # find the colors within the specified boundaries and apply
#         # the mask
#         mask = cv2.inRange(image, lower, upper)
#         output = cv2.bitwise_and(image, image, mask=mask)
#         tot_pix = count_nonblack_np(image)
#         color_pix = count_nonblack_np(output)
#         ratio = color_pix / tot_pix
#         #         print("ratio is:", ratio)
#         # if ratio > 0.01 and i == 0:
#         #     return 'red'
#         # elif ratio > 0.01 and i == 1:
#         #     return 'yellow'
#
#         i += 1
#
#         if show:
#             cv2.imshow("images", np.hstack([image, output]))
#             # if cv2.waitKey(0) & 0xFF == ord('q'):
#             #     cv2.destroyAllWindows()
#     return ratio
#     # return 'not_sure'

def get_team(persons):
    team_ls = []
    for person in persons:
        team_ls.append(detect_team(person))
    return team_ls


def detect_team(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    res, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
    cv2.moveWindow("input", 400, 500)

    ratio = detect_light_dark(thresh)

    if ratio > 0.7:
        return 1
    else:
        return 0


def count_black(img):
    cnt = 0
    for row in img:
        for pix in row:
            if pix == 255:
                cnt += 1
    return cnt


def detect_light_dark(img):
    total_pix = img.shape[0] * img.shape[1]
    color_pix = count_black(img)
    return color_pix/total_pix


def plot_bbox(img, bbox, conf, cls):
    height, width = img.shape[:2]
    if len(bbox) != 0:
        for i, box in enumerate(bbox):
            # get x1 x2 x3 x4
            x1 = int(round(((box[0] - box[2] / 2.0) * width).item()))
            y1 = int(round(((box[1] - box[3] / 2.0) * height).item()))
            x2 = int(round(((box[0] + box[2] / 2.0) * width).item()))
            y2 = int(round(((box[1] + box[3] / 2.0) * height).item()))
            # print(x1)
            cls_conf = conf[i]
            cls_id = cls[i]
            import random
            color = random.choices(range(256), k=3)
            color = [int(x) for x in np.random.randint(256, size=3)]
            # put texts and rectangles
            if cls_id == 0:
                img = cv2.putText(img, "person", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    return img


def get_person(img, bbox):
    height, width = img.shape[:2]
    person_ls = []
    for i, box in enumerate(bbox):
        # get x1 x2 x3 x4
        x1 = int(round(((box[0] - box[2] / 2.0) * width).item()))
        y1 = int(round(((box[1] - box[3] / 2.0) * height).item()))
        x2 = int(round(((box[0] + box[2] / 2.0) * width).item()))
        y2 = int(round(((box[1] + box[3] / 2.0) * height).item()))
        person_ls.append(np.asarray(img[y1: y2, x1: x2]))
    return person_ls


if __name__ == '__main__':
    img = cv2.imread("../frame/video1/video1_20.jpg")
