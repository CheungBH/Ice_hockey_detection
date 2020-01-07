import cv2
import os

step = 5


class Video2Frame(object):
    def __init__(self, folder="video3"):
        self.folder = folder
        self.video_ls = []
        self.save_folder = "../frame/{}".format(folder)
        os.makedirs(self.save_folder, exist_ok=True)

    def process_video(self, video):
        print("Processing video {}".format(video))
        name = video.split("\\")[-1]
        cap = cv2.VideoCapture(video)
        frm = 0
        while True:
            ret, frame = cap.read()
            frm += 1
            if ret:
                if frm % step == 0:
                    print("Saving the {} frame".format(frm))
                    cv2.imwrite(os.path.join(self.save_folder, "{}_{}.jpg".format(name[:-4], frm)), frame)
                else:
                    pass
            else:
                cv2.destroyAllWindows()
                break

    def process_folder(self):
        video_ls = [os.path.join(self.folder, video_path) for video_path in os.listdir(self.folder)]
        for v in video_ls:
            self.process_video(v)


if __name__ == '__main__':
    VF = Video2Frame()
    VF.process_video("..\\video\\video3.mp4")
