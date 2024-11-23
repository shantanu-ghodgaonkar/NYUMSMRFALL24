import os
import cv2 as cv
import numpy as np


class opticalFlow():
    def __init__(self, video_file_path: str = 'Task4/tracking.mp4', output_folder: str = 'output_frames'):
        # Params for ShiTomasi corner detection
        self.feature_params = dict(maxCorners=100,
                                   qualityLevel=0.3,
                                   minDistance=7,
                                   blockSize=7)
        # Parameters for lucas kanade optical flow
        self.lk_params = dict(winSize=(15, 15),
                              maxLevel=2,
                              criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
        # Create some random colors
        self.color = np.random.randint(0, 255, (100, 3))
        self.cap = cv.VideoCapture(video_file_path)
        self.output_folder = os.path.join(
            os.path.dirname(video_file_path), output_folder)
        # Ensure output folder exists
        os.makedirs(self.output_folder, exist_ok=True)
        # Frame rate of the video
        self.fps = int(self.cap.get(cv.CAP_PROP_FPS))

    def display_lk_optical_flow(self):
        ret, old_frame = self.cap.read()
        if not ret:
            print("Failed to read video!")
            return
        old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
        p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **self.feature_params)
        mask = np.zeros_like(old_frame)

        frame_counter = 0  # Initialize frame counter
        frame_interval = self.fps  # Save frames every 1 second

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print('No frames grabbed!')
                break
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            p1, st, err = cv.calcOpticalFlowPyrLK(
                old_gray, frame_gray, p0, None, **self.lk_params)
            if p1 is not None:
                good_new = p1[st == 1]
                good_old = p0[st == 1]

            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv.line(mask, (int(a), int(b)),
                               (int(c), int(d)), self.color[i].tolist(), 2)
                frame = cv.circle(frame, (int(a), int(b)), 5,
                                  self.color[i].tolist(), -1)
            img = cv.add(frame, mask)

            # Save frame every 1 second
            if frame_counter % frame_interval == 0:
                frame_path = os.path.join(
                    self.output_folder, f"lk_frame_{frame_counter}.png")
                cv.imwrite(frame_path, img)

            cv.imshow('frame', img)
            k = cv.waitKey(30) & 0xff
            if k == 27:  # Exit on 'ESC'
                break
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)
            frame_counter += 1  # Increment frame counter
        cv.destroyAllWindows()

    def display_dense_optical_flow(self):
        ret, frame1 = self.cap.read()
        if not ret:
            print("Failed to read video!")
            return
        prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame1)
        hsv[..., 1] = 255

        frame_counter = 0  # Initialize frame counter
        frame_interval = self.fps  # Save frames every 1 second

        while True:
            ret, frame2 = self.cap.read()
            if not ret:
                print('No frames grabbed!')
                break
            next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
            flow = cv.calcOpticalFlowFarneback(
                prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
            bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

            # Save frame every 1 second
            if frame_counter % frame_interval == 0:
                frame_path = os.path.join(
                    self.output_folder, f"dense_frame_{frame_counter}.png")
                cv.imwrite(frame_path, bgr)

            cv.imshow('frame2', bgr)
            k = cv.waitKey(30) & 0xff
            if k == 27:  # Exit on 'ESC'
                break
            prvs = next
            frame_counter += 1  # Increment frame counter
        cv.destroyAllWindows()


def main():
    of_object = opticalFlow()

    # Example to display and save frames for LK Optical Flow
    # of_object.display_lk_optical_flow()

    # Example to display and save frames for Dense Optical Flow
    # Uncomment the following line to test:
    of_object.display_dense_optical_flow()


if __name__ == '__main__':
    main()
