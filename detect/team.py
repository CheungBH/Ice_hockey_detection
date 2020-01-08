import numpy as np
import cv2

index_dict = {0: "white", 1: "blue"}
color2RGB = {"white": (255, 255, 255), "blue": (255, 0, 0), "red": (0, 0, 255)}


class TeamDetector(object):
    def __init__(self, img, bbox, conf, ids):
        self.img = img
        self.bbox = bbox
        self.person_img = []
        self.person = []
        self.team = []
        self.conf = conf
        self.id = ids

    def get_person(self):
        height, width = self.img.shape[:2]
        for i, box in enumerate(self.bbox):
            # get x1 x2 x3 x4
            x1 = int(round(((box[0] - box[2] / 2.0) * width).item()))
            y1 = int(round(((box[1] - box[3] / 2.0) * height).item()))
            x2 = int(round(((box[0] + box[2] / 2.0) * width).item()))
            y2 = int(round(((box[1] + box[3] / 2.0) * height).item()))
            self.person_img.append(np.asarray(self.img[y1: y2, x1: x2]))

    def get_team(self):
        for idx, self.person in enumerate(self.person_img):
            try:
                self.team.append(self.detect_team())
            except:
                self.team.append("red")

    def detect_team(self):
        gray = cv2.cvtColor(self.person, cv2.COLOR_BGR2GRAY)
        res, thresh = cv2.threshold(gray, 64, 255, cv2.THRESH_BINARY)
        ratio = self.detect_light_dark(thresh)
        if ratio > 0.85:
            return index_dict[0]
        else:
            return index_dict[1]

    def detect_light_dark(self, img):
        total_pix = img.shape[0] * img.shape[1]
        color_pix = self.count_black(img)
        return color_pix / total_pix

    @staticmethod
    def count_black(img):
        cnt = 0
        for row in img:
            for pix in row:
                if pix == 255:
                    cnt += 1
        return cnt

    def plot_info_with_team(self):
        height, width = self.img.shape[:2]
        for i, box in enumerate(self.bbox):
            color = color2RGB[self.team[i]]
            if self.id[i] == 0:
                # get x1 x2 x3 x4
                x1 = int(round(((box[0] - box[2] / 2.0) * width).item()))
                y1 = int(round(((box[1] - box[3] / 2.0) * height).item()))
                x2 = int(round(((box[0] + box[2] / 2.0) * width).item()))
                y2 = int(round(((box[1] + box[3] / 2.0) * height).item()))
                self.person_img.append(np.asarray(self.img[y1: y2, x1: x2]))
                self.img = cv2.putText(self.img, " ", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                self.img = cv2.rectangle(self.img, (x1, y1), (x2, y2), color, 2)

    def process(self):
        self.get_person()
        self.get_team()
        self.plot_info_with_team()
        return self.img



