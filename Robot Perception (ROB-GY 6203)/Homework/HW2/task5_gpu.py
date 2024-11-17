import os
import cv2
from cuml.cluster import KMeans as cuKMeans  # GPU-based KMeans from RAPIDS
import numpy as np
from tqdm import tqdm
from datetime import datetime
import pickle


class VPR:
    def __init__(self, method='BoVW', n_clusters=1000):
        """
        Initialize the pipeline.

        Args:
            method (str): Feature aggregation method ('BoVW' or 'VLAD').
            n_clusters (int): Number of clusters for visual vocabulary.
        """
        self.method = method
        self.n_clusters = n_clusters

        # Use GPU-accelerated ORB detector (replace SIFT)
        self.detector = cv2.cuda.SIFT_create()
        self.kmeans = None
        self.vocabulary = None

    def load_image_paths(self, directory):
        """
        Load all image paths from a directory.

        Args:
            directory (str): Path to the directory containing images.

        Returns:
            list: List of image file paths.
        """
        image_paths = [
            os.path.join(directory, filename)
            for filename in os.listdir(directory)
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
        ]
        return image_paths

    def extract_features(self, image_paths):
        """
        Extract feature descriptors from all images with a progress bar.

        Args:
            image_paths (list): List of image file paths.

        Returns:
            list: List of feature descriptors for all images.
        """
        descriptors_list = []
        for path in tqdm(image_paths, desc="Extracting Features"):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # Upload the image to the GPU
                gpu_img = cv2.cuda_GpuMat()
                gpu_img.upload(img)

                # Detect and compute descriptors on GPU
                keypoints, descriptors = self.detector.detectAndComputeAsync(
                    gpu_img, None)
                if descriptors is not None:
                    descriptors_list.extend(descriptors)
        return descriptors_list

    def train_vocabulary(self, descriptors_list):
        """
        Train the visual vocabulary using GPU-accelerated K-Means clustering.

        Args:
            descriptors_list (list): List of feature descriptors.
        """
        print("Training K-Means on GPU (this may take time)...")

        # Convert descriptors list to a numpy array
        descriptors = np.array(descriptors_list, dtype=np.float32)

        # Use RAPIDS cuML KMeans for GPU-based clustering
        self.kmeans = cuKMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            verbose=1
        )
        self.kmeans.fit(descriptors)
        self.vocabulary = self.kmeans.cluster_centers_

    def compute_vlad(self, descriptors):
        """
        Compute VLAD representation for a single image.

        Args:
            descriptors (np.array): Feature descriptors for the image.

        Returns:
            np.array: VLAD vector for the image.
        """
        k = self.kmeans.n_clusters
        vlad_vector = np.zeros((k, descriptors.shape[1]), dtype=np.float32)

        for descriptor in tqdm(descriptors, desc='Computing VLAD'):
            idx = self.kmeans.predict(descriptor.reshape(1, -1))
            residual = descriptor - self.vocabulary[idx]
            vlad_vector[idx] += residual

        vlad_vector = vlad_vector.flatten()
        # L2 normalization
        return vlad_vector / np.linalg.norm(vlad_vector, ord=2)

    def describe_images(self, image_paths):
        """
        Generate image descriptions using the chosen method (BoVW or VLAD) with a progress bar.

        Args:
            image_paths (list): List of image file paths.

        Returns:
            list: List of image descriptions.
        """
        image_descriptions = []
        for path in tqdm(image_paths, desc=f"Describing Images using {self.method}"):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                gpu_img = cv2.cuda_GpuMat()
                gpu_img.upload(img)
                keypoints, descriptors = self.detector.detectAndComputeAsync(
                    gpu_img, None)
                if descriptors is not None:
                    if self.method == 'BoVW':
                        words = self.kmeans.predict(descriptors)
                        histogram, _ = np.histogram(
                            words, bins=range(self.kmeans.n_clusters + 1))
                        image_descriptions.append(histogram)
                    elif self.method == 'VLAD':
                        vlad_vector = self.compute_vlad(descriptors)
                        image_descriptions.append(vlad_vector)
        return image_descriptions

    def run_pipeline(self, directory):
        """
        Execute the entire pipeline for a directory of images.

        Args:
            directory (str): Path to the directory containing images.

        Returns:
            list: List of image descriptions.
        """
        print(f"Loading images from: {directory}")
        self.image_paths = self.load_image_paths(directory)
        print(f"Found {len(self.image_paths)} images.")

        # Extract features
        descriptors_list = self.extract_features(self.image_paths)

        # Train visual vocabulary
        self.train_vocabulary(descriptors_list)

        # Describe images
        self.image_descriptions = self.describe_images(self.image_paths)

        print(f"Pipeline complete! Saving results.")
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.save_descriptions_to_numpy(
            output_file=f'VPR_{self.method}/img_desc_{timestamp}.npz')
        self.save_vocabulary(
            output_file=f'VPR_{self.method}/vocabulary_{timestamp}.pkl')

    def save_descriptions_to_numpy(self, output_file="image_descriptions.npz"):
        """
        Save image paths and descriptions as a compressed NumPy file.

        Args:
            output_file (str): Path to save the NumPy file.
        """
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}")
        np.savez(output_file, image_paths=self.image_paths,
                 image_descriptions=self.image_descriptions)
        print(f"Descriptions saved to {output_file}")

    def save_vocabulary(self, output_file="vocabulary.pkl"):
        """
        Save the KMeans vocabulary to a file.

        Args:
            output_file (str): Path to save the vocabulary.
        """
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}")
        with open(output_file, "wb") as f:
            pickle.dump(self.kmeans, f)
        print(f"Vocabulary saved to {output_file}")


def main():
    directory = "images"
    pipeline = VPR(method='VLAD', n_clusters=1000)
    pipeline.run_pipeline(directory=directory)


if __name__ == '__main__':
    main()
