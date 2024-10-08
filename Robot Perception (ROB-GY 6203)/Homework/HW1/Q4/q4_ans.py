import cv2
import numpy as np
from pathlib import Path
from glob import glob

# Define the folder containing images
FOLDER_PATH = Path(f"./Q4/img").absolute()  # Absolute path to the image folder
SAVE_PATH = FOLDER_PATH / "Processed_imgs"   # Path to save processed images

# Create the save folder if it doesn't exist
SAVE_PATH.mkdir(parents=True, exist_ok=True)

# Camera intrinsic matrix K generated through OpenCV Code
K = np.array([
    [432.15077202,   0.0,        376.86467699],
    [0.0,        451.18127325, 258.30382425],
    [0.0,          0.0,          1.0]
])

# Alternative Camera intrinsic matrix K generated through MATLAB Camera Calibration Tool
# Uncomment the following lines to use the MATLAB-generated matrix
# K = np.array([
#     [472.465976443917,	0,	328.564352614446],
#     [0,	512.211333600681,	261.888069867386],
#     [0,	0,	1]
# ])

# Define the marker length in meters (adjust based on your actual marker size)
MARKER_LENGTH = 0.1  # 100 mm


def load_images(folder_path):
    """
    Load all PNG images from the specified folder.

    Args:
        folder_path (Path): Path to the folder containing images.

    Returns:
        tuple:
            list: List of successfully loaded images as NumPy arrays.
            list: Corresponding image file paths for the loaded images.
    """
    # Retrieve all PNG image paths in the folder
    image_paths = glob(f'{folder_path}/*.png')

    # Read each image using OpenCV
    images = [cv2.imread(path) for path in image_paths]

    # Lists to store successfully loaded images and their paths
    loaded_images = []
    valid_paths = []

    # Iterate through the images to verify successful loading
    for idx, img in enumerate(images):
        if img is not None:
            loaded_images.append(img)
            valid_paths.append(image_paths[idx])
        else:
            print(f"Error loading image {image_paths[idx]}")

    return loaded_images, valid_paths


def initialize_aruco_detector():
    """
    Initialize the ArUco detector with a predefined dictionary and parameters.

    Returns:
        tuple:
            cv2.aruco_ArucoDetector: Initialized ArUco detector object.
            cv2.aruco_Dictionary: ArUco dictionary used for detection.
    """
    # Define the type of ArUco dictionary to use (DICT_5X5_100)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)

    # Set detector parameters (using default parameters)
    parameters = cv2.aruco.DetectorParameters()

    # Create the ArUco detector with the specified dictionary and parameters
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    return detector, aruco_dict


def estimate_pose(corners, marker_length, camera_matrix):
    """
    Estimate the pose (rotation and translation vectors) of detected ArUco markers.

    Args:
        corners (list): List of detected marker corners.
        marker_length (float): The actual length of the marker's side (in meters).
        camera_matrix (numpy.ndarray): Camera intrinsic matrix.

    Returns:
        tuple:
            numpy.ndarray: Rotation vectors for each detected marker.
            numpy.ndarray: Translation vectors for each detected marker.
    """
    # Estimate pose of each detected marker using solvePnP
    # Assuming zero distortion coefficients
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
        corners=corners,
        markerLength=marker_length,
        cameraMatrix=camera_matrix,
        distCoeffs=np.zeros((4, 1))  # Zero distortion coefficients
    )
    return rvecs, tvecs


def draw_cube(img, rvec, tvec, camera_matrix, marker_length):
    """
    Draw a 3D cube on the detected ArUco marker within the image.

    Args:
        img (numpy.ndarray): The original image on which to draw the cube.
        rvec (numpy.ndarray): Rotation vector of the marker.
        tvec (numpy.ndarray): Translation vector of the marker.
        camera_matrix (numpy.ndarray): Camera intrinsic matrix.
        marker_length (float): The actual length of the marker's side (in meters).

    Returns:
        numpy.ndarray: The image with the 3D cube drawn on the marker.
    """
    # Define the cube's 3D coordinates in the marker's coordinate system
    # The marker is assumed to lie on the XY plane with Z=0
    # The cube will have the same base size as the marker and height equal to marker_length
    half_length = marker_length / 2
    cube_points = np.array([
        [-half_length, -half_length, 0],
        [half_length, -half_length, 0],
        [half_length,  half_length, 0],
        [-half_length,  half_length, 0],
        [-half_length, -half_length, marker_length],
        [half_length, -half_length, marker_length],
        [half_length,  half_length, marker_length],
        [-half_length,  half_length, marker_length]
    ], dtype=np.float32)

    # Project the 3D cube points to the 2D image plane
    imgpts, _ = cv2.projectPoints(
        cube_points,
        rvec,
        tvec,
        camera_matrix,
        None  # Assuming no lens distortion
    )
    imgpts = np.int32(imgpts).reshape(-1, 2)  # Convert to integer coordinates

    # Draw the base of the cube (the marker)
    img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), 3)

    # Draw the vertical pillars of the cube
    for i in range(4):
        img = cv2.line(img, tuple(imgpts[i]), tuple(
            imgpts[i + 4]), (255, 0, 0), 3)

    # Draw the top of the cube
    img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)

    return img


def main():
    """
    Main function to process images: detect ArUco markers, estimate their pose, draw 3D cubes,
    and save the processed images.
    """
    # Load images from the specified folder
    images, image_paths = load_images(FOLDER_PATH)

    # Check if any images were loaded successfully
    if not images:
        raise ValueError(
            "No images were loaded. Please check the folder path and image files."
        )

    # Initialize the ArUco detector
    detector, _ = initialize_aruco_detector()

    # List to store processed images
    processed_images = []

    # Iterate through each loaded image
    for idx, img in enumerate(images):
        print(f"Processing image {idx + 1}/{len(images)}: {image_paths[idx]}")

        # Convert the image to grayscale for marker detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect ArUco markers in the grayscale image
        corners, ids, rejected = detector.detectMarkers(gray)

        # Check if any markers were detected
        if ids is not None:
            print(f"Detected markers: {ids.flatten()}")

            # Estimate the pose (rotation and translation) for each detected marker
            rvecs, tvecs = estimate_pose(
                corners=corners,
                marker_length=MARKER_LENGTH,
                camera_matrix=K
            )

            # Iterate through each detected marker to draw annotations
            for i in range(len(ids)):
                # Draw the detected marker boundaries and ID
                img = cv2.aruco.drawDetectedMarkers(
                    image=img,
                    corners=[corners[i]],
                    ids=np.array([ids[i]])
                )

                # Draw a 3D cube on the marker
                img = draw_cube(
                    img=img,
                    rvec=rvecs[i],
                    tvec=tvecs[i],
                    camera_matrix=K,
                    marker_length=MARKER_LENGTH
                )
        else:
            print("No markers detected.")

        # Append the processed image to the list
        processed_images.append(img)

        # Display the processed image in a window
        cv2.imshow(f'AR Result - Image {idx + 1}', img)

        # Wait indefinitely until a key is pressed
        cv2.waitKey(0)

        # Prepare the filename for the processed image
        original_filename = Path(image_paths[idx]).name
        processed_filename = f"processed_{original_filename}"
        output_file = SAVE_PATH / processed_filename  # Path for saving the image

        # Save the processed image to the designated folder
        cv2.imwrite(str(output_file), img)
        print(f"Saved processed image to {output_file}")

        # Close the image display window
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
