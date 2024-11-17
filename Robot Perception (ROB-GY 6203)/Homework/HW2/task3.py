import numpy as np
import cv2 as cv
import glob
from typing import List, Tuple


def calibrate_camera(
    display_input_imgs: bool = False,
    verbose: bool = False,
    img_path: str = 'Task3/[0-9]*.jpg',
    chessboard_size: Tuple[int, int] = (8, 6)
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
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((chessboard_size[1] * chessboard_size[0], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0],
                           0:chessboard_size[1]].T.reshape(-1, 2)
    objpoints, imgpoints = [], []

    images = glob.glob(img_path)
    if not images:
        raise ValueError(f"No images found at path: {img_path}")
    print(f"Found {len(images)} images for calibration.")

    for fname in images:
        img = cv.imread(fname)
        if img is None:
            print(f"Warning: Unable to read image {fname}. Skipping.")
            continue
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, chessboard_size, None)

        if ret:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            if display_input_imgs:
                cv.drawChessboardCorners(img, chessboard_size, corners2, ret)
                cv.imshow('img', img)
                cv.waitKey(500)
        else:
            if verbose:
                print(f"Chessboard corners not found in image {fname}.")

    cv.destroyAllWindows()

    if not objpoints or not imgpoints:
        raise ValueError("No valid chessboard corners detected in any image.")

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )
    if verbose:
        print(f"RMS Re-projection error: {ret}")
        print(f"Camera Matrix:\n{mtx}")
        print(f"Distortion Coefficients:\n{dist}")
    return ret, mtx, dist, rvecs, tvecs


def validate_F_matrix(F: np.ndarray, pts1: np.ndarray, pts2: np.ndarray) -> None:
    """
    Validate the Fundamental Matrix by checking the epipolar constraint.

    Parameters:
    - F (np.ndarray): Fundamental matrix.
    - pts1 (np.ndarray): Points from the left image.
    - pts2 (np.ndarray): Points from the right image.
    """
    # Convert points to homogeneous coordinates
    pts1_h = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
    pts2_h = np.hstack((pts2, np.ones((pts2.shape[0], 1))))

    # Compute the epipolar constraint
    epipolar_error = np.abs(np.sum(pts2_h @ F * pts1_h, axis=1))
    print(f"Epipolar Constraint Mean Error (F): {np.mean(epipolar_error):.6f}")
    print(f"Epipolar Constraint Max Error (F): {np.max(epipolar_error):.6f}")


def validate_E_matrix(E: np.ndarray, K: np.ndarray, pts1: np.ndarray, pts2: np.ndarray) -> None:
    """
    Validate the Essential Matrix by checking its rank and epipolar constraint in normalized space.

    Parameters:
    - E (np.ndarray): Essential matrix.
    - K (np.ndarray): Camera intrinsic matrix.
    - pts1 (np.ndarray): Points from the left image.
    - pts2 (np.ndarray): Points from the right image.
    """
    # Normalize points using the intrinsic matrix
    pts1_h = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
    pts2_h = np.hstack((pts2, np.ones((pts2.shape[0], 1))))
    pts1_normalized = (np.linalg.inv(K) @ pts1_h.T).T
    pts2_normalized = (np.linalg.inv(K) @ pts2_h.T).T

    # Compute the epipolar constraint
    epipolar_error = np.abs(
        np.sum(pts2_normalized @ E * pts1_normalized, axis=1))
    print(f"Epipolar Constraint Mean Error (E): {np.mean(epipolar_error):.6f}")
    print(f"Epipolar Constraint Max Error (E): {np.max(epipolar_error):.6f}")

    # Validate rank-2 constraint
    U, S, Vt = np.linalg.svd(E)
    print(f"Singular Values of E: {S}")


def validate_pose(R: np.ndarray, t: np.ndarray, pts1: np.ndarray, pts2: np.ndarray, K: np.ndarray) -> None:
    """
    Validate the recovered pose by triangulating 3D points and ensuring positive depth.

    Parameters:
    - R (np.ndarray): Rotation matrix.
    - t (np.ndarray): Translation vector.
    - pts1 (np.ndarray): Points from the left image.
    - pts2 (np.ndarray): Points from the right image.
    - K (np.ndarray): Camera intrinsic matrix.
    """
    # Projection matrices
    proj_matrix1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    proj_matrix2 = K @ np.hstack((R, t.reshape(-1, 1)))

    # Triangulate points
    points_4D = cv.triangulatePoints(
        proj_matrix1, proj_matrix2, pts1.T, pts2.T)
    points_3D = (points_4D[:3] / points_4D[3]).T

    # Check for positive depth relative to both cameras
    depth1 = points_3D[:, 2]  # Depth relative to first camera
    points_cam2 = (R @ points_3D.T + t.reshape(3, 1)).T
    depth2 = points_cam2[:, 2]

    valid_points = np.logical_and(depth1 > 0, depth2 > 0)
    print(
        f"Percentage of points with positive depth: {np.mean(valid_points) * 100:.2f}%")

    # Compute reprojection errors
    points_proj1_h = proj_matrix1 @ points_4D
    points_proj1 = (points_proj1_h[:2] / points_proj1_h[2]).T
    points_proj2_h = proj_matrix2 @ points_4D
    points_proj2 = (points_proj2_h[:2] / points_proj2_h[2]).T

    error1 = np.linalg.norm(pts1 - points_proj1, axis=1)
    error2 = np.linalg.norm(pts2 - points_proj2, axis=1)

    print(
        f"Reprojection Error in Image 1: Mean={np.mean(error1):.4f}, Max={np.max(error1):.4f}")
    print(
        f"Reprojection Error in Image 2: Mean={np.mean(error2):.4f}, Max={np.max(error2):.4f}")


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
    img_left = cv.imread(left_img_path)
    img_right = cv.imread(right_img_path)
    if img_left is None or img_right is None:
        raise ValueError("One or both images could not be loaded.")

    gray_left = cv.cvtColor(img_left, cv.COLOR_BGR2GRAY)
    gray_right = cv.cvtColor(img_right, cv.COLOR_BGR2GRAY)

    aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)
    parameters = cv.aruco.DetectorParameters()

    corners_left, ids_left, _ = cv.aruco.detectMarkers(
        gray_left, aruco_dict, parameters=parameters)
    corners_right, ids_right, _ = cv.aruco.detectMarkers(
        gray_right, aruco_dict, parameters=parameters)

    if ids_left is None or ids_right is None:
        raise ValueError("Aruco markers not detected in one or both images.")

    common_ids = np.intersect1d(ids_left, ids_right)
    if len(common_ids) == 0:
        raise ValueError("No common Aruco markers found between the images.")

    points_left, points_right = [], []

    for common_id in common_ids:
        idx_left = np.where(ids_left == common_id)[0][0]
        idx_right = np.where(ids_right == common_id)[0][0]
        # Use the corners of the marker for correspondence
        corners_marker_left = corners_left[idx_left][0]
        corners_marker_right = corners_right[idx_right][0]
        for corner_l, corner_r in zip(corners_marker_left, corners_marker_right):
            points_left.append(corner_l)
            points_right.append(corner_r)

    points_left = np.array(points_left)
    points_right = np.array(points_right)

    if len(points_left) < 8:
        raise ValueError(
            "Not enough correspondences to compute the Fundamental Matrix.")

    F, mask = cv.findFundamentalMat(points_left, points_right, cv.FM_RANSAC,
                                    ransacReprojThreshold=1.0, confidence=0.99)
    if verbose:
        print(f"Fundamental Matrix:\n{F}")

    # Optionally display epipolar lines
    if display_epipolar_lines and F is not None:
        def draw_epipolar_lines(img1, img2, lines, pts1, pts2):
            """ Draw epipolar lines on the images """
            r, c, _ = img1.shape
            for r_line, pt1, pt2 in zip(lines, pts1, pts2):
                color = (0, 255, 0)
                x0, y0 = map(int, [0, -r_line[2]/r_line[1]])
                x1, y1 = map(int, [c, -(r_line[2] + r_line[0]*c)/r_line[1]])
                img1 = cv.line(img1, (x0, y0), (x1, y1), color, 2)
                img1 = cv.circle(img1, tuple(pt1.astype(int)), 5, color, -2)
                img2 = cv.circle(img2, tuple(pt2.astype(int)), 5, color, -2)
            return img1, img2

        # Select inlier points
        inlier_pts1 = points_left[mask.ravel() == 1]
        inlier_pts2 = points_right[mask.ravel() == 1]

        # Compute epilines in the right image for points in the left image
        lines1 = cv.computeCorrespondEpilines(
            inlier_pts2.reshape(-1, 1, 2), 2, F)
        lines1 = lines1.reshape(-1, 3)
        img5, img6 = draw_epipolar_lines(
            img_left.copy(), img_right.copy(), lines1, inlier_pts1, inlier_pts2)

        # Compute epilines in the left image for points in the right image
        lines2 = cv.computeCorrespondEpilines(
            inlier_pts1.reshape(-1, 1, 2), 1, F)
        lines2 = lines2.reshape(-1, 3)
        img3, img4 = draw_epipolar_lines(
            img_right.copy(), img_left.copy(), lines2, inlier_pts2, inlier_pts1)

        # Save the images with epipolar lines
        cv.imwrite('Task3/Left_Image_Epilines.jpg', img5)
        cv.imwrite('Task3/Right_Image_Epilines.jpg', img3)

        # Display the images with epipolar lines
        cv.imshow('Left Image Epilines', img5)
        cv.imshow('Right Image Epilines', img3)
        cv.waitKey(0)
        cv.destroyAllWindows()

    return F, points_left, points_right


def estimate_E_and_pose(
    K: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate the essential matrix and recover the relative pose (R, t).

    Parameters:
    - K (np.ndarray): Camera intrinsic matrix.
    - pts1 (np.ndarray): Points from the left image.
    - pts2 (np.ndarray): Points from the right image.
    - verbose (bool): If True, prints the essential matrix and pose.

    Returns:
    - Tuple[np.ndarray, np.ndarray, np.ndarray]:
        - Essential matrix (E)
        - Rotation matrix (R)
        - Translation vector (t)
    """
    # Convert points to normalized coordinates
    pts1_norm = cv.undistortPoints(np.expand_dims(pts1, axis=1), K, None)
    pts2_norm = cv.undistortPoints(np.expand_dims(pts2, axis=1), K, None)

    # Estimate the Essential Matrix
    E, mask = cv.findEssentialMat(
        pts1_norm, pts2_norm, method=cv.RANSAC, prob=0.999, threshold=1.0)
    if E is None:
        raise ValueError("Essential matrix could not be estimated.")
    if verbose:
        print(f"Essential Matrix:\n{E}")

    # Recover pose (R, t) from the Essential Matrix
    _, R, t, mask_pose = cv.recoverPose(E, pts1_norm, pts2_norm, mask=mask)
    if verbose:
        print("Rotation Matrix (R):\n", R)
        print("Translation Vector (t):\n", t)

    # Normalize the translation vector to unit length
    t = t / np.linalg.norm(t)
    if verbose:
        print("Normalized Translation Vector (t):\n", t)

    return E, R, t


def test_all_poses(E: np.ndarray, K: np.ndarray, pts1: np.ndarray, pts2: np.ndarray) -> None:
    """
    Decompose the Essential Matrix into possible poses and evaluate each.

    Parameters:
    - E (np.ndarray): Essential matrix.
    - K (np.ndarray): Camera intrinsic matrix.
    - pts1 (np.ndarray): Points from the left image.
    - pts2 (np.ndarray): Points from the right image.
    """
    # Decompose E into possible R and t
    U, S, Vt = np.linalg.svd(E)
    if np.linalg.det(U) < 0:
        U *= -1.0
    if np.linalg.det(Vt) < 0:
        Vt *= -1.0

    W = np.array([[0, -1, 0],
                  [1,  0, 0],
                  [0,  0, 1]])

    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t = U[:, 2]

    # Ensure R is a valid rotation matrix
    if np.linalg.det(R1) < 0:
        R1 = -R1
    if np.linalg.det(R2) < 0:
        R2 = -R2

    # Define possible poses
    poses = [(R1,  t),
             (R1, -t),
             (R2,  t),
             (R2, -t)]

    # Convert points to normalized coordinates
    pts1_norm = cv.undistortPoints(np.expand_dims(
        pts1, axis=1), K, None).reshape(-1, 2).T
    pts2_norm = cv.undistortPoints(np.expand_dims(
        pts2, axis=1), K, None).reshape(-1, 2).T

    # Camera 1 projection matrix
    proj_matrix1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    proj_matrix1 = K @ proj_matrix1

    best_pose = None
    max_positive_depth = 0
    best_reproj_error = np.inf

    for i, (R, t_vec) in enumerate(poses):
        # Projection matrix for camera 2
        proj_matrix2 = np.hstack((R, t_vec.reshape(-1, 1)))
        proj_matrix2 = K @ proj_matrix2

        # Triangulate points
        points_4D = cv.triangulatePoints(
            proj_matrix1, proj_matrix2, pts1_norm[:2], pts2_norm[:2])
        points_4D /= points_4D[3]  # Normalize homogeneous coordinates
        points_3D = points_4D[:3].T

        # Check for positive depth
        depth1 = points_3D[:, 2]
        points_cam2 = (R @ points_3D.T + t_vec.reshape(3, 1)).T
        depth2 = points_cam2[:, 2]
        valid_points = np.logical_and(depth1 > 0, depth2 > 0)
        num_positive = np.sum(valid_points)

        # Compute reprojection errors only for valid points
        if num_positive > 0:
            # Reproject points onto both images
            points_proj1_h = proj_matrix1 @ points_4D
            points_proj1 = (points_proj1_h[:2] / points_proj1_h[2]).T

            points_proj2_h = proj_matrix2 @ points_4D
            points_proj2 = (points_proj2_h[:2] / points_proj2_h[2]).T

            # Compute reprojection errors
            error1 = np.linalg.norm(pts1 - points_proj1, axis=1)
            error2 = np.linalg.norm(pts2 - points_proj2, axis=1)
            mean_error = np.mean(error1 + error2)

            # Update the best pose based on positive depth and reprojection error
            if num_positive > max_positive_depth or (num_positive == max_positive_depth and mean_error < best_reproj_error):
                max_positive_depth = num_positive
                best_reproj_error = mean_error
                best_pose = (R, t_vec)

        print(f"Pose {i + 1}:")
        print(f"  Rotation Matrix (R):\n{R}")
        print(f"  Translation Vector (t):\n{t_vec}")
        print(f"  Positive Depth Points: {num_positive} / {len(pts1)}")
        print(
            f"  Mean Reprojection Error: {mean_error if num_positive > 0 else 'N/A'}\n")

    if best_pose is not None:
        print("Best Pose Selected:")
        print(f"Rotation Matrix (R):\n{best_pose[0]}")
        print(f"Translation Vector (t):\n{best_pose[1]}")
    else:
        print("No valid pose found with positive depth for any points.")


def main():
    try:
        # Camera calibration
        ret, K, dist, rvecs, tvecs = calibrate_camera(verbose=True)

        # Load and compute correspondences
        F, pts1, pts2 = estimate_F_mtx(
            verbose=True, display_epipolar_lines=True)

        # Validate Fundamental Matrix
        validate_F_matrix(F, pts1, pts2)

        # Estimate Essential Matrix and pose using cv.findEssentialMat
        E, R, t = estimate_E_and_pose(K, pts1, pts2, verbose=True)

        # Validate Essential Matrix
        validate_E_matrix(E, K, pts1, pts2)

        # Validate the recovered pose
        validate_pose(R, t, pts1, pts2, K)

        # Test all possible poses
        # test_all_poses(E=E, K=K, pts1=pts1, pts2=pts2)

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    main()
