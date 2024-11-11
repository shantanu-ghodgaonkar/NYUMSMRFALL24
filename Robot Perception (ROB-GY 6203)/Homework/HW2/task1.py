import numpy as np
import open3d as o3d
from tqdm import tqdm


class RANSAC:
    """
    A class to perform RANSAC for plane fitting on a set of 3D points.

    Attributes:
        best_inliers (list): Indices of the points that are inliers for the best plane found.
        best_plane (np.ndarray or None): Coefficients of the best plane equation [a, b, c, d].
        d_t (float): Distance threshold to determine if a point is an inlier.
    """

    def __init__(self, points: np.ndarray, distance_threshold: float):
        """
        Initializes the RANSAC class and performs RANSAC to find the best plane.

        Args:
            points (np.ndarray): Array of 3D points.
            distance_threshold (float): Distance threshold to classify points as inliers.
        """
        np.set_printoptions(precision=3)
        self.best_inliers = []
        self.best_plane = None
        self.d_t = distance_threshold
        print(
            f'RANSAC started for given points with distance threshold = {self.d_t}')
        self.ransac(points=points, distance_threshold=self.d_t)
        print(
            f'Best plane found to be {self.best_plane} at distance threshold of {self.d_t}')

    def calculate_plane(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
        """
        Calculates the plane equation from three points.

        Args:
            p1 (np.ndarray): First point.
            p2 (np.ndarray): Second point.
            p3 (np.ndarray): Third point.

        Returns:
            np.ndarray: Coefficients of the plane equation [a, b, c, d], or None if points are collinear.
        """
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm == 0:
            return None  # Collinear points; cannot define a plane
        normal = normal / norm
        d = -np.dot(normal, p1)
        return np.array([normal[0], normal[1], normal[2], d])

    def point_to_plane_distance(self, point: np.ndarray, plane: np.ndarray) -> float:
        """
        Calculates the perpendicular distance of a point from a plane.

        Args:
            point (np.ndarray): 3D point to measure distance from.
            plane (np.ndarray): Plane coefficients [a, b, c, d].

        Returns:
            float: Perpendicular distance of the point from the plane.
        """
        return (abs((plane[0] * point[0]) + (plane[1] * point[1]) + (plane[2] * point[2]) + plane[3]) /
                np.sqrt((plane[0]**2) + (plane[1]**2) + (plane[2]**2)))

    def ransac(self, points: np.ndarray, MAX_ITER: int = 1000, distance_threshold: float = 0.01):
        """
        Performs the RANSAC algorithm to find the best plane fitting a set of 3D points.

        Args:
            points (np.ndarray): Array of 3D points.
            MAX_ITER (int, optional): Maximum number of RANSAC iterations. Defaults to 1000.
            distance_threshold (float, optional): Distance threshold for inliers. Defaults to 0.01.
        """
        np.random.seed(42)
        N = np.inf  # Initially set to infinity to start adaptive sampling
        sample_count = 0  # Track the number of iterations
        with tqdm(total=MAX_ITER, desc='RANSAC Progress') as pbar:
            while ((sample_count < N) and (sample_count < MAX_ITER)):
                inliers = []  # Reset inliers for each iteration
                # Randomly sample 3 points to define a plane
                sample_points = points[np.random.choice(
                    range(len(points)), 3, replace=False)]
                plane = self.calculate_plane(
                    p1=sample_points[0], p2=sample_points[1], p3=sample_points[2])
                if plane is None:
                    sample_count += 1
                    pbar.update(1)
                    continue  # Skip iteration if plane cannot be defined

                # Calculate inliers based on the distance threshold
                for i, point in enumerate(points):
                    if self.point_to_plane_distance(point=point, plane=plane) < distance_threshold:
                        inliers.append(i)

                # Update the best model if the current one has more inliers
                if len(inliers) > len(self.best_inliers):
                    self.best_inliers = inliers
                    self.best_plane = plane

                    # Estimate the outlier ratio and calculate required iterations (N) for 99% success probability
                    epsilon = (1 - (len(inliers) / len(points)))
                    try:
                        N = np.log(1 - 0.99) / np.log(1 - ((1 - epsilon)**3))
                        N = int(np.ceil(N))
                        # Update total iterations if N is less than MAX_ITER
                        if N < MAX_ITER:
                            pbar.total = N
                    except (RuntimeWarning, ZeroDivisionError, ValueError):
                        pass
                sample_count += 1
                pbar.update(1)


def main():
    # read demo point cloud provided by Open3D
    pcd_point_cloud = o3d.data.PCDPointCloud()
    pcd = o3d.io.read_point_cloud(pcd_point_cloud.path)

    best_ransac_object = RANSAC(points=np.asarray(pcd.points),
                                distance_threshold=0.027)

    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name=f"Point Cloud with RANSAC Plane Fitting using Distance Threshold = {best_ransac_object.d_t}")
    # Separate inliers and outliers
    inlier_cloud = pcd.select_by_index(
        best_ransac_object.best_inliers)
    outlier_cloud = pcd.select_by_index(
        best_ransac_object.best_inliers, invert=True)

    # Visualize the results
    inlier_cloud.paint_uniform_color([1.0, 0, 0])  # Color inliers red
    outlier_cloud.paint_uniform_color([0, 1, 0])   # Color outliers green
    vis.add_geometry(inlier_cloud)
    vis.add_geometry(outlier_cloud)

    # Run visualization
    vis.run()
    vis.destroy_window()


if __name__ == '__main__':
    main()
