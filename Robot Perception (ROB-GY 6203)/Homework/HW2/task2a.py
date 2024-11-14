import open3d as o3d
import copy
import numpy as np
from tqdm import tqdm
from datetime import datetime

demo_icp_pcds = o3d.data.DemoICPPointClouds()
source = o3d.io.read_point_cloud(demo_icp_pcds.paths[0])
target = o3d.io.read_point_cloud(demo_icp_pcds.paths[1])


class ICP:
    """
    A class to perform the Iterative Closest Point (ICP) algorithm for point cloud alignment.
    """

    def __init__(self, src_pts: np.ndarray, tgt_pts: np.ndarray, MAX_ITER: int = 1000, tol: float = 1e-4):
        """
        Initializes the ICP algorithm with source and target points.

        :param src_pts: Source points as a NumPy array of shape (N, 3).
        :param tgt_pts: Target points as a NumPy array of shape (M, 3).
        :param MAX_ITER: Maximum number of iterations for ICP.
        :param tol: Convergence tolerance for error reduction.
        """
        self.src_pts = src_pts
        self.tgt_pts = tgt_pts
        self.transformation = self.icp(MAX_ITER=MAX_ITER, tol=tol)

    def find_closest_pts(self) -> np.ndarray:
        """
        Finds the closest points in the target point cloud for each point in the source point cloud.

        :return: An array of indices representing the closest points in the target point cloud.
        """
        # Convert target points to Open3D point cloud format
        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(self.tgt_pts)
        tgt_kdtree = o3d.geometry.KDTreeFlann(target_pcd)

        # For each source point, find the closest point in the target
        correspondences = []
        for point in self.src_pts:
            # Perform nearest neighbor search; treat point as a 1D array for search
            _, idx, _ = tgt_kdtree.search_knn_vector_3d(point, 1)
            correspondences.append(idx[0])

        return np.array(correspondences)

    def compute_transformation(self, correspondences: np.ndarray) -> tuple:
        """
        Computes the optimal rotation and translation to align the source points with the target points.

        :param correspondences: Array of indices representing the closest points in the target point cloud.
        :return: A tuple (R, t) where R is the rotation matrix and t is the translation vector.
        """
        # Center the source and target points
        src_centered = self.src_pts - np.mean(self.src_pts, axis=0)
        tgt_centered = self.tgt_pts[correspondences] - \
            np.mean(self.tgt_pts[correspondences], axis=0)

        # Compute the cross-covariance matrix
        H = np.dot(src_centered.T, tgt_centered)
        U, _, Vt = np.linalg.svd(H)

        # Calculate rotation matrix
        R = np.dot(Vt.T, U.T)

        # Ensure that R is a proper rotation matrix by checking its determinant
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = np.dot(Vt.T, U.T)

        # Calculate translation vector
        t = np.mean(self.tgt_pts[correspondences], axis=0) - \
            np.dot(R, np.mean(self.src_pts, axis=0))
        return R, t

    def icp(self, MAX_ITER: int, tol: float) -> np.ndarray:
        """
        Runs the ICP algorithm to align the source point cloud with the target point cloud.

        :param MAX_ITER: Maximum number of iterations for ICP.
        :param tol: Convergence tolerance for error reduction.
        :return: The final transformation matrix as a 4x4 NumPy array.
        """
        transformation = np.eye(4)  # Initialize transformation as identity
        error = np.inf  # Initialize error as infinity

        # Run ICP iterations with progress bar
        with tqdm(range(MAX_ITER), desc=f"ICP Progress - Error: {error:.6f}") as pbar:
            for i in pbar:
                # Find closest points and compute transformation
                correspondences = self.find_closest_pts()
                R, t = self.compute_transformation(
                    correspondences=correspondences)
                # Apply transformation to source points
                transformed_src_pts = np.dot(self.src_pts, R) + t

                # Calculate the alignment error for this iteration
                error_i = np.mean(np.linalg.norm(
                    self.tgt_pts[correspondences] - transformed_src_pts, axis=1))

                # Check if the cost decreased; if not, stop early
                if error_i >= error:
                    pbar.set_description(
                        f"ICP Progress - Error: {error_i:.6f}")
                    pass

                # Apply the transformation only if error decreased
                self.src_pts = transformed_src_pts
                transformation_i = np.eye(4)
                transformation_i[:3, :3] = R
                transformation_i[:3, 3] = t
                transformation = np.dot(transformation_i, transformation)

                # Check for convergence based on tolerance
                if abs(error - error_i) < tol:
                    print("Convergence reached; stopping early.")
                    break

                # Update error and progress bar
                error = error_i
                pbar.set_description(f"ICP Progress - Error: {error_i:.6f}")

        # Save the final transformation matrix to file
        np.save(
            f"task2a_transformations/transformation_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}_{tol}.npy", transformation)
        return transformation


def draw_registration_result(source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud, transformation: np.ndarray):
    """
    Visualizes the alignment of the source and target point clouds using the provided transformation.

    :param source: The source point cloud.
    :param target: The target point cloud.
    :param transformation: The 4x4 transformation matrix to apply to the source point cloud.
    """
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])  # Color source in orange
    target_temp.paint_uniform_color([0, 0.651, 0.929])  # Color target in blue
    source_temp.transform(transformation)  # Apply transformation to source
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])


def main():
    """
    Main function to execute the ICP algorithm and visualize the results.
    """
    draw_registration_result(source=source, target=target,
                             transformation=np.eye(4))
    icp_object = ICP(src_pts=np.asarray(source.points),
                     tgt_pts=np.asarray(target.points))
    draw_registration_result(source=source, target=target,
                             transformation=icp_object.transformation)


if __name__ == '__main__':
    main()
