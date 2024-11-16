import numpy as np
import cv2 as cv
import glob
from typing import List, Tuple, Optional


def calibrate_camera(
    display_input_imgs: bool = False,
    verbose: bool = False,
    img_path: str = 'Task3/[0-9]*.jpg',
    chessboard_size: Tuple[int, int] = (7, 6)  # Default to 7x6 chessboard
) -> Tuple[float, np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]:
    """
    Calibrate the camera using a set of chessboard images.

    Parameters:
    - display_input_imgs (bool): If True, displays input images with detected chessboard corners.
    - verbose (bool): If True, prints calibration details.
    - img_path (str): Path pattern to chessboard images.
    - chessboard_size (Tuple[int, int]): Number of inner corners per row and column of the chessboard.

    Returns:
    - Tuple[float, np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]:
        - RMS Re-projection error
        - Camera matrix (intrinsic parameters)
        - Distortion coefficients
        - List of rotation vectors (extrinsic parameters)
        - List of translation vectors (extrinsic parameters)
    """
    # Termination criteria for corner refinement
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare 3D object points (chessboard coordinates)
    objp = np.zeros((chessboard_size[1] * chessboard_size[0], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0],
                           0:chessboard_size[1]].T.reshape(-1, 2)

    # Storage for object points and image points
    objpoints = []  # 3D points in the real world
    imgpoints = []  # 2D points in the image plane

    # Load all images matching the pattern
    images = glob.glob(img_path)

    images = glob.glob(img_path)
    if not images:
        raise ValueError(f"No images found at path: {img_path}")
    else:
        print(f"Found {len(images)} images for calibration.")

    if not images:
        raise ValueError(f"No images found at path: {img_path}")

    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv.findChessboardCorners(gray, chessboard_size, None)

        if ret:  # If corners found
            objpoints.append(objp)

            # Refine corner locations
            corners2 = cv.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            if display_input_imgs:
                # Display the corners on the image
                cv.drawChessboardCorners(img, chessboard_size, corners2, ret)
                cv.imshow('img', img)
                cv.waitKey(500)

    cv.destroyAllWindows()

    # Calibrate the camera
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    if verbose:
        print(f"\nRMS Re-projection error: {ret}\n")
        print(f"Camera Matrix:\n{mtx}\n")
        print(f"Distortion Coefficients:\n{dist}\n")
        print(f"Rotation Vectors:\n{rvecs}\n")
        print(f"Translation Vectors:\n{tvecs}\n")

    return ret, mtx, dist, rvecs, tvecs


def draw_epipolar_lines(
    img: np.ndarray, lines: np.ndarray, points: np.ndarray, color: Tuple[int, int, int]
) -> np.ndarray:
    """
    Draw epipolar lines on the image.

    Parameters:
    - img (np.ndarray): Image on which epipolar lines will be drawn.
    - lines (np.ndarray): Epipolar lines.
    - points (np.ndarray): Points corresponding to the epipolar lines.
    - color (Tuple[int, int, int]): Color of the epipolar lines.

    Returns:
    - np.ndarray: Image with epipolar lines drawn.
    """
    h, w = img.shape[:2]
    img = img.copy()
    for r, pt in zip(lines, points):
        # Compute endpoints of the epipolar line
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [w, -(r[2] + r[0] * w) / r[1]])
        img = cv.line(img, (x0, y0), (x1, y1), color, 1)
        img = cv.circle(img, (int(pt[0]), int(pt[1])), 5, color, -1)
    return img


def estimate_F_mtx(
    display_epipolar_lines: bool = False,
    verbose: bool = False,
    left_img_path: str = 'Task3/left.jpg',
    right_img_path: str = 'Task3/right.jpg'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate the fundamental matrix and optionally display epipolar lines.

    Parameters:
    - display_epipolar_lines (bool): If True, displays epipolar lines on the images.
    - verbose (bool): If True, prints the fundamental matrix.
    - left_img_path (str): Path to the left image.
    - right_img_path (str): Path to the right image.

    Returns:
    - Tuple[np.ndarray, np.ndarray, np.ndarray]:
        - Fundamental matrix (F)
        - Corresponding points from the left image
        - Corresponding points from the right image
    """
    # Load and preprocess images
    img_left = cv.imread(left_img_path)
    img_right = cv.imread(right_img_path)
    gray_left = cv.cvtColor(img_left, cv.COLOR_BGR2GRAY)
    gray_right = cv.cvtColor(img_right, cv.COLOR_BGR2GRAY)

    # Detect Aruco markers
    aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)
    parameters = cv.aruco.DetectorParameters()

    corners_left, ids_left, _ = cv.aruco.detectMarkers(
        gray_left, aruco_dict, parameters=parameters)
    corners_right, ids_right, _ = cv.aruco.detectMarkers(
        gray_right, aruco_dict, parameters=parameters)

    if ids_left is None or ids_right is None:
        raise ValueError("Aruco markers not detected in one or both images.")

    # Match marker IDs and extract corresponding points
    common_ids = np.intersect1d(ids_left, ids_right)
    points_left, points_right = [], []

    for common_id in common_ids:
        idx_left = np.where(ids_left == common_id)[0][0]
        idx_right = np.where(ids_right == common_id)[0][0]

        for i in range(4):  # Each marker has 4 corners
            points_left.append(corners_left[idx_left][0][i])
            points_right.append(corners_right[idx_right][0][i])

    points_left = np.array(points_left)
    points_right = np.array(points_right)

    # Compute the fundamental matrix
    F, mask = cv.findFundamentalMat(points_left, points_right, cv.FM_LMEDS)

    if verbose:
        print(f"Fundamental Matrix:\n{F}")

    if display_epipolar_lines:
        # Draw and display epipolar lines
        lines_left = cv.computeCorrespondEpilines(
            points_right.reshape(-1, 1, 2), 2, F).reshape(-1, 3)
        img_left_with_lines = draw_epipolar_lines(
            img_left, lines_left, points_left, (255, 0, 0))

        lines_right = cv.computeCorrespondEpilines(
            points_left.reshape(-1, 1, 2), 1, F).reshape(-1, 3)
        img_right_with_lines = draw_epipolar_lines(
            img_right, lines_right, points_right, (255, 0, 0))

        scale_factor = 0.5
        cv.imshow("Left Image with Epipolar Lines", cv.resize(img_left_with_lines,
                  None, fx=scale_factor, fy=scale_factor, interpolation=cv.INTER_AREA))
        cv.imshow("Right Image with Epipolar Lines", cv.resize(img_right_with_lines,
                  None, fx=scale_factor, fy=scale_factor, interpolation=cv.INTER_AREA))
        cv.waitKey(0)
        cv.destroyAllWindows()

    return F, points_left, points_right


def main() -> None:
    """
    Main function to calibrate the camera, estimate the fundamental matrix,
    and recover the pose from the essential matrix.
    """
    _, K, dist, rvecs, tvecs = calibrate_camera(verbose=False)
    F, pts1, pts2 = estimate_F_mtx(verbose=False, display_epipolar_lines=False)

    # Compute Essential Matrix
    E = K.T @ F @ K

    # Recover pose from the Essential Matrix
    _, R, t, mask = cv.recoverPose(E, pts1, pts2, K)

    print("Essential Matrix:\n", E)
    print("Rotation Matrix:\n", R)
    print("Translation Vector:\n", t)


if __name__ == '__main__':
    main()
