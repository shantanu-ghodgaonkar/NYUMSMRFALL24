import numpy as np
import cv2 as cv
import glob
from pathlib import Path
from time import time, sleep

# Constants
K = 'k2'  # 'k1' for images within 1m and 'k2' for images within 2m and 3m distance
CAM_PORT = 2  # Camera port index
FOLDER_PATH = Path(f"./Q3/img_{K}")  # Path to save captured images


def capture_images(folder_path, cam_port, num_images=3, delay_initial=5, delay_between=3):
    """
    Capture images from the specified camera port and save them to the designated folder.

    Args:
        folder_path (Path): Directory where captured images will be saved.
        cam_port (int): Index of the camera port to capture images from.
        num_images (int, optional): Number of images to capture. Defaults to 3.
        delay_initial (int, optional): Initial delay before starting image capture (in seconds). Defaults to 5.
        delay_between (int, optional): Delay between consecutive image captures (in seconds). Defaults to 3.

    Raises:
        Exception: If the camera cannot be accessed.
    """
    # Ensure the save directory exists
    folder_path.mkdir(parents=True, exist_ok=True)

    print(
        f"Initial delay of {delay_initial} seconds before starting image capture to get the pose ready...")
    sleep(delay_initial)

    # Initialize the video capture object
    cap = cv.VideoCapture(cam_port)

    if not cap.isOpened():
        raise Exception(f"Cannot open camera with port {cam_port}")

    for i in range(num_images):
        print(f"Capturing image {i + 1}/{num_images}...")
        ret, img = cap.read()

        if ret:
            timestamp = int(time())
            img_filename = folder_path / f"img{timestamp}.jpg"
            cv.imwrite(str(img_filename), img)
            print(f"Image saved to {img_filename}")
        else:
            print("No image detected.")

        sleep(delay_between)

    # Release the camera resource
    cap.release()
    print("Image capture completed.")


def find_chessboard_corners(folder_path, chessboard_size=(9, 7), criteria=None):
    """
    Detect chessboard corners in all images within the specified folder.

    Args:
        folder_path (Path): Directory containing images to process.
        chessboard_size (tuple, optional): Number of inner corners per a chessboard row and column. Defaults to (9, 7).
        criteria (tuple, optional): Termination criteria for corner refinement. Defaults to None.

    Returns:
        tuple:
            list: Object points in real-world space.
            list: Image points in image plane.
            list: Filenames of images where corners were detected.
    """
    if criteria is None:
        # Default termination criteria: either 30 iterations or move by at least 0.001
        criteria = (cv.TERM_CRITERIA_EPS +
                    cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points based on the real chessboard dimensions
    objp = np.zeros((chessboard_size[1] * chessboard_size[0], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0],
                           0:chessboard_size[1]].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images
    objpoints = []  # 3D points in real-world space
    imgpoints = []  # 2D points in image plane
    successful_images = []  # Filenames of successfully processed images

    # Retrieve all JPEG images in the folder
    images = glob.glob(str(folder_path / '*.jpg'))

    print(f"Found {len(images)} images for chessboard detection.")

    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv.findChessboardCorners(gray, chessboard_size, None)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)

            # Refine corner locations to sub-pixel accuracy
            corners2 = cv.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            successful_images.append(fname)

            # Draw and display the corners
            cv.drawChessboardCorners(img, chessboard_size, corners2, ret)
            cv.imshow('Chessboard Corners', img)
            cv.waitKey(500)  # Display each image for 500ms
            print(
                f"Chessboard detected and corners refined for image: {fname}")
        else:
            print(f"Chessboard not detected in image: {fname}")

    cv.destroyAllWindows()
    print(
        f"Chessboard detection completed. {len(successful_images)} out of {len(images)} images were successful.")

    return objpoints, imgpoints, successful_images


def calibrate_camera(objpoints, imgpoints, image_shape):
    """
    Perform camera calibration to compute the camera matrix and distortion coefficients.

    Args:
        objpoints (list): 3D points in real-world space.
        imgpoints (list): 2D points in image plane.
        image_shape (tuple): Shape of the calibration images (height, width).

    Returns:
        tuple:
            float: RMS re-projection error.
            numpy.ndarray: Camera matrix (intrinsic parameters).
            numpy.ndarray: Distortion coefficients.
            list: Rotation vectors.
            list: Translation vectors.
    """
    print("Starting camera calibration...")
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, image_shape[::-1], None, None
    )
    print("Camera calibration completed.")
    return ret, mtx, dist, rvecs, tvecs


def calculate_mean_reprojection_error(objpoints, imgpoints, rvecs, tvecs, camera_matrix, dist_coeffs):
    """
    Calculate the mean re-projection error for the calibration.

    Args:
        objpoints (list): 3D points in real-world space.
        imgpoints (list): 2D points in image plane.
        rvecs (list): Rotation vectors from calibration.
        tvecs (list): Translation vectors from calibration.
        camera_matrix (numpy.ndarray): Camera intrinsic matrix.
        dist_coeffs (numpy.ndarray): Distortion coefficients.

    Returns:
        float: Mean re-projection error.
    """
    mean_error = 0
    total_points = 0

    for i in range(len(objpoints)):
        # Project the 3D object points to 2D image points
        imgpoints2, _ = cv.projectPoints(
            objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
        )

        # Compute the Euclidean distance between detected and projected points
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        mean_error += error
        total_points += 1

    mean_error /= total_points
    return mean_error


def main():
    """
    Main function to execute the camera calibration process:
    1. Capture images from the camera.
    2. Detect chessboard corners in the captured images.
    3. Calibrate the camera using the detected corners.
    4. Calculate and display the mean re-projection error.
    """
    # Step 1: Capture images from the camera | Uncomment below line if you want to capture new images
    # capture_images(folder_path=FOLDER_PATH, cam_port=CAM_PORT, num_images=15)

    # Step 2: Detect chessboard corners in the captured images
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objpoints, imgpoints, successful_images = find_chessboard_corners(
        folder_path=FOLDER_PATH,
        chessboard_size=(9, 7),
        criteria=criteria
    )

    if not objpoints or not imgpoints:
        print(
            "No chessboard corners were detected in any image. Calibration cannot proceed.")
        return

    # Assume all images have the same shape; use the first successful image to get the shape
    sample_img = cv.imread(successful_images[0])
    gray_sample = cv.cvtColor(sample_img, cv.COLOR_BGR2GRAY)
    image_shape = gray_sample.shape

    # Step 3: Calibrate the camera
    ret, mtx, dist, rvecs, tvecs = calibrate_camera(
        objpoints=objpoints,
        imgpoints=imgpoints,
        image_shape=image_shape
    )

    # Step 4: Calculate the mean re-projection error
    mean_error = calculate_mean_reprojection_error(
        objpoints=objpoints,
        imgpoints=imgpoints,
        rvecs=rvecs,
        tvecs=tvecs,
        camera_matrix=mtx,
        dist_coeffs=dist
    )

    print(f"Total mean re-projection error: {mean_error}")
    print(f"Camera Matrix ({K}):\n{mtx}")


if __name__ == "__main__":
    main()
