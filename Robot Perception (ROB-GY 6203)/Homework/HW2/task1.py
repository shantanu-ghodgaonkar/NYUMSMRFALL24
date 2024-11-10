import numpy as np
import open3d as o3d
from tqdm import tqdm  # Import tqdm for progress bar


class RANSAC():
    def __init__(self, points: np.ndarray, distance_threshold):
        np.set_printoptions(precision=3)
        self.best_inliers = []
        self.best_plane = None
        self.d_t = distance_threshold
        self.ransac(points=points, distance_threshold=self.d_t)

        print(
            f'Best plane found to be {self.best_plane} at distance threshold of {self.d_t}')

    def calculate_plane(self, p1, p2, p3):
        v1 = p2 - p1
        v2 = p3 - p1

        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm == 0:
            return None  # Colinear points; cannot define a plane
        normal = normal / norm

        d = -np.dot(normal, p1)
        return np.array([normal[0], normal[1], normal[2], d])

    def point_to_plane_distance(self, point, plane):
        return (abs((plane[0] * point[0]) + (plane[1] * point[1]) + (plane[2] * point[2]) + plane[3]) /
                np.sqrt((plane[0]**2) + (plane[1]**2) + (plane[2]**2)))

    def ransac(self, points: np.ndarray, MAX_ITER: int = 1000, distance_threshold: float = 0.01):
        np.random.seed(42)
        N = np.inf
        sample_count = 0
        with tqdm(total=MAX_ITER, desc='RANSAC Progress') as pbar:
            while ((sample_count < N) and (sample_count < MAX_ITER)):
                inliers = []
                # Sample without replacement
                sample_points = points[np.random.choice(
                    range(len(points)), 3, replace=False)]
                plane = self.calculate_plane(
                    p1=sample_points[0], p2=sample_points[1], p3=sample_points[2])
                if plane is None:
                    sample_count += 1
                    pbar.update(1)
                    continue  # Skip iteration if plane cannot be defined

                for i, point in enumerate(points):
                    if self.point_to_plane_distance(point=point, plane=plane) < distance_threshold:
                        inliers.append(i)

                if len(inliers) > len(self.best_inliers):
                    self.best_inliers = inliers
                    self.best_plane = plane

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

    # distance_threshold_list = [0.001, 0.005, 0.01, 0.05]
    distance_threshold_list = np.linspace(0.01428, 0.027, 15)
    ransac_objects = [RANSAC(points=np.asarray(pcd.points),
                             distance_threshold=d_t) for d_t in distance_threshold_list]
    for ransac_object in ransac_objects:

        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name=f"Point Cloud with RANSAC Plane Fitting using Distance Threshold = {ransac_object.d_t}")
        # Separate inliers and outliers
        inlier_cloud = pcd.select_by_index(
            ransac_object.best_inliers)
        outlier_cloud = pcd.select_by_index(
            ransac_object.best_inliers, invert=True)

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
