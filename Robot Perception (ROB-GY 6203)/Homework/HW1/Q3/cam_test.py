# program to capture single image from webcam in python

# importing OpenCV library
from cv2 import VideoCapture, imshow, imwrite, waitKey, destroyWindow
from pathlib import Path

# initialize the camera
# If you have multiple camera connected with
# current device, assign a value in cam_port
# variable according to that
folder_path = Path(f"./Q3/img")
cam_port = 2
cam = VideoCapture(cam_port)

# reading the input using the camera
result, image = cam.read()

# If image will detected without any error,
# show result
if result:

    # showing result, it take frame name and image
    # output
    imshow("ArucoTag", image)

    # saving image in local storage
    imwrite(f"{folder_path.absolute()}/ArucoTag_3.png", image)

    # If keyboard interrupt occurs, destroy image
    # window
    waitKey(0)
    destroyWindow("ArucoTag")

# If captured image is corrupted, moving to else part
else:
    print("No image detected. Please! try again")
