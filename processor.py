import numpy as np
import time
import cv2
# import pylab
import os
import sys
from sklearn.decomposition import FastICA
from jade import jadeR
from scipy.signal import butter, lfilter


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


class faceTracking(object):

    def __init__(self, buffer_size=250):
        self.gray = 0

        self.frame_in = np.zeros((10, 10))
        self.frame_out = np.zeros((10, 10))
        self.fps = 0

        self.slices = [[0]]
        self.t0 = time.time()

        # dpath = resource_path("haarcascade_frontalface_alt.xml")
        # if not os.path.exists(dpath):
        #     print("Cascade file not present!")
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades
                                                  + 'haarcascade_frontalface_default.xml')
        # self.cap = cv2.VideoCapture(0)

        self.data_buffer, self.times, = [], []
        self.buffer_size = buffer_size
        self.samples = []

        self.face_rect = [1, 1, 2, 2]
        self.rect_size = (0.5, 0.15, 0.3, 0.15)
        # self.rect_size = (0.5, 0.15, 0.05, 0.05)
        self.last_center = np.array([0, 0])
        self.last_wh = np.array([0, 0])
        self.output_dim = 13
        self.trained = False

        self.idx = 1
        self.find_faces = True

        self.fft = [[], [], []]
        self.bpms = [[], [], []]
        self.freqs = []

        self.last_peak = None
        self.diff = 10

    def find_faces_toggle(self):
        self.find_faces = not self.find_faces
        return self.find_faces

    def get_subface_coord(self, rect_size):  # find forehead coordinate with relative position
        x, y, w, h = self.face_rect
        fh_x, fh_y, fh_w, fh_h = rect_size
        return [int(x + w * fh_x - (w * fh_w / 2.0)),
                int(y + h * fh_y - (h * fh_h / 2.0)),
                int(w * fh_w),
                int(h * fh_h)]

    def getPixelMean(self, coord):  # means of input frame for each color chanel, with light equalization
        x, y, w, h = coord
        subframe = self.frame_in[y:y + h, x:x + w, :]
        img_hsv = cv2.cvtColor(subframe, cv2.COLOR_RGB2HSV)
        img_hsv[:, :, 2] = cv2.equalizeHist(img_hsv[:, :, 2])
        image2 = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
        mask = cv2.inRange(image2,
                           lowerb=np.array([50, 50, 50], dtype="uint8"),
                           upperb=np.array([255, 255, 255], dtype="uint8"))
        # image2 = image2[np.where(mask > 0)]
        # v1 = np.mean(image2[:, 0])  # RGB values of subframe
        # v2 = np.mean(image2[:, 1])
        # v3 = np.mean(image2[:, 2])
        v1 = np.mean(image2[:, :, 0])  # RGB values of subframe
        v2 = np.mean(image2[:, :, 1])
        v3 = np.mean(image2[:, :, 2])

        # return (v1 + v2 + v3) / 3.  # mean of rgb value
        return [v1, v2, v3]

    def draw_rect(self, rect, col=(0, 255, 0)):
        x, y, w, h = rect
        cv2.rectangle(self.frame_out, (x, y), (x + w, y + h), col, 1)

    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y

    def run(self):
        self.times.append(time.time() - self.t0)
        self.frame_out = self.frame_in

        self.gray = cv2.equalizeHist(cv2.cvtColor(self.frame_in,
                                                  cv2.COLOR_BGR2GRAY))
        col = (100, 255, 100)
        if self.find_faces:

            # put texts on window
            cv2.putText(
                self.frame_out, "Press 'S' to lock face and begin",
                (10, 50), cv2.FONT_HERSHEY_PLAIN, 1.25, col)
            cv2.putText(self.frame_out, "Press 'Esc' to quit",
                        (10, 75), cv2.FONT_HERSHEY_PLAIN, 1.25, col)
            # initiating data
            self.data_buffer, self.times, self.trained = [], [], False
            # image = self.frame_in
            # cv2.namedWindow("equalization", cv2.WINDOW_GUI_NORMAL)
            # # convert image from RGB to HSV
            # img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            # # Histogram equalisation on the V-channel
            # img_hsv[:, :, 2] = cv2.equalizeHist(img_hsv[:, :, 2])
            # # convert image back from HSV to RGB
            # image2 = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
            # stack = np.hstack((image, image2))
            # cv2.imshow("lightness", image2)

            # Detect the faces
            faces = list(self.face_cascade.detectMultiScale(self.gray,
                                                            scaleFactor=1.3,
                                                            minNeighbors=4,
                                                            minSize=(50, 50),
                                                            flags=cv2.CASCADE_SCALE_IMAGE))
            # Draw the rectangle around each face

            if len(faces) > 0:
                faces.sort(key=lambda a: a[-1] * a[-2])
                self.face_rect = faces[-1]
            # roi = self.face_rect
            roi = self.get_subface_coord(self.rect_size)
            self.draw_rect(self.face_rect, col=(255, 0, 0))
            self.draw_rect(roi)
            x, y, w, h = self.face_rect
            cv2.putText(self.frame_out, "Face",
                        (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, col)
            return
            # for (x, y, w, h) in faces:
            #     cv2.rectangle(self.frame_in, (x, y), (x + w, y + h), (255, 0, 0), 1)
        if set(self.face_rect) == set([1, 1, 2, 2]):
            return
        cv2.putText(
            self.frame_out, "Press 'S' to restart",
            (10, 50), cv2.FONT_HERSHEY_PLAIN, 1.5, col)
        cv2.putText(self.frame_out, "Press 'D' to toggle data plot",
                    (10, 75), cv2.FONT_HERSHEY_PLAIN, 1.5, col)
        cv2.putText(self.frame_out, "Press 'F' to save data",
                    (10, 100), cv2.FONT_HERSHEY_PLAIN, 1.5, col)
        cv2.putText(self.frame_out, "Press 'Esc' to quit",
                    (10, 125), cv2.FONT_HERSHEY_PLAIN, 1.5, col)
        roi = self.get_subface_coord(self.rect_size)
        # roi = self.face_rect
        self.draw_rect(roi)
        # while True:
        #     # Read the frame
        #     #_, img = self.cap.read()
        #     # Convert to grayscale
        #     # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #     self.gray = cv2.equalizeHist(cv2.cvtColor(self.frame_in,
        #                                               cv2.COLOR_BGR2GRAY))
        #     # Detect the faces
        #     faces = self.face_cascade.detectMultiScale(self.gray,
        #                                                scaleFactor=1.3,
        #                                                minNeighbors=4,
        #                                                minSize=(50, 50),
        #                                                flags=cv2.CASCADE_SCALE_IMAGE)
        #     # Draw the rectangle around each face
        #     for (x, y, w, h) in faces:
        #         cv2.rectangle(self.frame_in, (x, y), (x + w, y + h), (255, 0, 0), 1)
        #     # Display
        #     #cv2.imshow("Processed", self.frame_in)
        #     # Stop if escape key is pressed
        #     # k = cv2.waitKey(10) & 255
        #     # if k == 27:
        #     #     print("Exiting")
        #     #     sys.exit()

        # Release the VideoCapture object
        # self.cap.release()
        pixel_vals = self.getPixelMean(roi)
        # print(pixel_vals)

        self.data_buffer.append(pixel_vals)
        L = len(self.data_buffer)
        if L > self.buffer_size:
            self.data_buffer = self.data_buffer[-self.buffer_size:]
            self.times = self.times[-self.buffer_size:]
            L = self.buffer_size
        processed = np.array(self.data_buffer)
        self.samples = np.transpose(processed)
        # np.transpose(self.samples)
        # print(self.samples.shape)

        if L > 10:
            x1, y1, w1, h1 = self.face_rect
            self.slices = [np.copy(self.frame_out[y1:y1 + h1, x1:x1 + w1, 1])]
            # print(len(self.samples))

            # transformer = FastICA(n_components=3,
            #                       random_state=0, max_iter=1000, tol=1)
            # X_transformed = transformer.fit_transform(self.samples)

            X_transformed = jadeR(self.samples, m=3, verbose=False)
            X_ = np.matmul(np.linalg.inv(X_transformed), self.samples)
            X_ = np.array(X_)

            for i in range(len(self.samples)):
                # color = X_[i]
                color = self.samples[i]
                # y = self.butter_bandpass_filter(color, 0.8, 6, 100, 4)
                even_times = np.linspace(self.times[0], self.times[-1], L)
                interpolated = np.interp(even_times, self.times, color)
                interpolated = np.hamming(L) * interpolated  # a wave with width L * interpolated value
                interpolated = interpolated - np.mean(interpolated)  # standardisation?
                raw = np.fft.rfft(interpolated)
                arg = np.abs(raw)
                self.freqs = np.fft.rfftfreq(L) * 10 * 60
                idx = np.where((self.freqs > 50) & (self.freqs < 160))

                # select the one within (50, 160)
                self.freqs = self.freqs[idx]
                self.fft[i] = arg[idx]
                # find the argmax
                peak = np.argmax(self.fft[i])
                if not self.last_peak:
                    self.last_peak = peak
                if (self.freqs[peak] - self.freqs[self.last_peak]) <= self.diff:
                    self.last_peak = peak
            self.bpms[i].append(self.freqs[self.last_peak])

            x1, y1, w1, h1 = self.face_rect
            self.slices = [np.copy(self.frame_out[y1:y1 + h1, x1:x1 + w1, 1])]
            col = (100, 255, 100)
            # gap = (self.buffer_size - L) / self.fps
            # if gap:
            #     text = "(estimate: %0.1f bpm, wait %0.0f s)" % (self.bpm, gap)
            # else:
            #     text = "(estimate: %0.1f bpm)" % (np.mean(np.transpose(self.bpm)[-1]))
            text = "(estimate: %0.1f bpm)" % (np.mean(np.transpose(self.bpms)[-1]))

            tsize = 1
            x, y, w, h = self.get_subface_coord(self.rect_size)
            cv2.putText(self.frame_out, text,
                        (int(x - w / 2), int(y)), cv2.FONT_HERSHEY_PLAIN, tsize, col)
            # detected2 = list(self.face_cascade.detectMultiScale(self.gray,
            #                                                     scaleFactor=1.3,
            #                                                     minNeighbors=4,
            #                                                     minSize=(
            #                                                         50, 50),
            #                                                     flags=cv2.CASCADE_SCALE_IMAGE))

            # if len(detected2) > 0:
            #     # print(len(detected2))
            #     detected2.sort(key=lambda a: a[-1] * a[-2])
            #
            #     if self.shift(detected2[-1]) > 0:
            #         face_rect2 = detected2[-1]
            #         self.face_detected = True
            #
            #     x, y, w, h = self.face_rect
            #     center0 = np.array([x + 0.5 * w, y + 0.5 * h])
            #     x, y, w, h = detected2[-1]
            #     center1 = np.array([x + 0.5 * w, y + 0.5 * h])
            #     shift1 = np.linalg.norm(center1 - center0)
            #     if shift1 > 10:
            #         self.face_detected = False
            #
            # else:
            #     self.face_detected = False
            # if self.face_detected:
            #     self.draw_rect(face_rect2, col=(255, 0, 0))
            #     if len(self.data_buffer) % self.save_freq == 0:
            #         print(sum(self.bpms[-self.save_freq:]) / self.save_freq, time.ctime())
            # else:
            #     cv2.putText(self.frame_out, str(self.face_detected),
            #                 (10, 150), cv2.FONT_HERSHEY_PLAIN, 1.5, col)

# if __name__ == "__main__":
#     while True:
#         face = faceTracking()
#         face.run()
