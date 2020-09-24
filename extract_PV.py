import cv2
import numpy as np
import sys
from device import Camera
from processor import faceTracking
from interface import plotXY, moveWindow
import datetime
from scipy.signal import butter, lfilter



class estimateHR(object):

    def __init__(self):

        self.cameras = []
        self.selected_cam = 0
        for i in range(3):
            camera = Camera(camera=i)  # first camera by default
            if camera.valid or not len(self.cameras):
                self.cameras.append(camera)
            else:
                break

        self.w, self.h = 0, 0
        self.pressed = 0

        self.processor = faceTracking(250)

        self.bpm_plot = False
        self.plot_title = "Data display - raw signal (top) and PSD (bottom)"

        self.key_controls = {"s": self.toggle_search,
                             "d": self.toggle_display_plot,
                             # "c": self.toggle_cam,
                             "f": self.write_csv}

    def toggle_search(self):
        """
        Toggles a motion lock on the processor's face detection component.

        Locking the forehead location in place significantly improves
        data quality, once a forehead has been sucessfully isolated.
        """
        state = self.processor.find_faces_toggle()
        print("face detection lock =", not state)

    def make_bpm_plot(self):
        """
        Creates and/or updates the data display
        """
        # plotXY([[self.processor.freqs,
        #          self.processor.fft[0]],
        #         [self.processor.freqs,
        #          self.processor.fft[1]],
        #         [self.processor.freqs,
        #          self.processor.fft[2]]
        #         ],
        #        labels=[True, True, True],
        #        showmax=["red", "green", "blue"],
        #        label_ndigits=[0, 0, 0],
        #        showmax_digits=[0, 0, 0],
        #        skip=[3, 3, 3],
        #        name="RGB pixel values",
        #        bg=self.processor.slices[0])

        # plotXY([[self.processor.freqs,
        #          self.processor.samples[0]],
        #         [self.processor.freqs,
        #          self.processor.samples[1]],
        #         [self.processor.freqs,
        #          self.processor.samples[2]]
        #         ],
        #        labels=[True, True, True],
        #        showmax=[False, False, False],
        #        label_ndigits=[0, 0, 0],
        #        showmax_digits=[0, 0, 0],
        #        skip=[3, 3, 3],
        #        name="RGB pixel values",
        #        bg=self.processor.slices[0])

        plotXY([[self.processor.times,
                 # np.mean(self.processor.samples, axis=0)],
                 self.processor.samples[1]],
                [self.processor.freqs,
                 # np.mean(self.processor.fft, axis=0)]],
                 self.processor.fft[1]]],
                 labels=[False, True],
               showmax=[False, "bpm"],
               label_ndigits=[0, 0],
               showmax_digits=[0, 1],
               skip=[3, 3],
               name="RGB pixel mean values",
               bg=self.processor.slices[0])



    def toggle_display_plot(self):
        """
        Toggles the data display.
        """
        if self.bpm_plot:
            print("bpm plot disabled")
            self.bpm_plot = False
            cv2.destroyWindow(self.plot_title)
        else:
            print("bpm plot enabled")
            if self.processor.find_faces:
                print("face found")
                self.toggle_search()
            # print("not found")
            self.bpm_plot = True
            self.make_bpm_plot()
            # print('ploting...')
            moveWindow(self.plot_title, self.w, 0)

    def write_csv(self):
        """
        Writes current data to a csv file
        """
        fn = "Webcam-pulse" + str(datetime.datetime.now())
        fn = fn.replace(":", "_").replace(".", "_")
        data_sample = np.vstack((self.processor.times, self.processor.samples)).T
        np.savetxt(fn + ".csv", data_sample, delimiter=",")
        # data_sample = np.vstack((self.processor.times, self.processor.samples)).T
        # k = min(len(self.processor.bpms), len(self.processor.times))
        # data = np.vstack((self.processor.times[-k:], self.processor.bpms[-k:])).T
        # np.savetxt(fn + "_raw.csv", data_sample, delimiter=",")
        # np.savetxt(fn + ".csv", data, delimiter=',')
        print("Writing csv")

    def key_handler(self):
        self.pressed = cv2.waitKey(10) & 255  # wait for keypress for 10 ms
        if self.pressed == 27:  # exit program on 'esc'
            print("Exiting")
            sys.exit()

        for key in self.key_controls.keys():
            if chr(self.pressed) == key:
                self.key_controls[key]()

    def main_loop(self):
        # Get current image frame from the camera
        frame = self.cameras[self.selected_cam].get_frame()
        self.h, self.w, _c = frame.shape

        # set current image frame to the processor's input
        self.processor.frame_in = frame
        # process the image frame to perform all needed analysis
        self.processor.run()
        # collect the output frame for display
        output_frame = self.processor.frame_out

        cv2.imshow("Processed", output_frame)

        if self.bpm_plot:
            self.make_bpm_plot()

        self.key_handler()


if __name__ == "__main__":
    app = estimateHR()
    while True:
        app.main_loop()
