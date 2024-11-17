import os
import cv2
from sklearn.cluster import KMeans
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
        # You can replace with ORB, SURF, etc.
        self.detector = cv2.SIFT_create()
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
                keypoints, descriptors = self.detector.detectAndCompute(
                    img, None)
                if descriptors is not None:
                    descriptors_list.extend(descriptors)
        return descriptors_list

    def train_vocabulary(self, descriptors_list):
        """
        Train the visual vocabulary using K-Means clustering with a progress bar.

        Args:
            descriptors_list (list): List of feature descriptors.
        """
        from tqdm import trange
        print("Training K-Means (this may take time)...")

        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            verbose=0
        )
        self.kmeans.fit(tqdm(descriptors_list, desc="Clustering Descriptors"))
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
            idx = self.kmeans.predict([descriptor])[0]
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
                keypoints, descriptors = self.detector.detectAndCompute(
                    img, None)
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

        # print("Extracting features...")
        descriptors_list = self.extract_features(self.image_paths)

        # print("Training visual vocabulary...")
        self.train_vocabulary(descriptors_list)

        # print(f"Describing images using {self.method}...")
        self.image_descriptions = self.describe_images(self.image_paths)

        print(f"Pipeline complete! Saving to file task_5/VPR_{self.method}.")
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.save_descriptions_to_numpy(
            output_file=f'task_5/VPR_{self.method}/img_desc_{timestamp}.npz')
        self.save_vocabulary(
            output_file=f'task_5/VPR_{self.method}/vocabulary_{timestamp}.pkl')

    def save_descriptions_to_numpy(self, output_file="image_descriptions.npz"):
        """
        Save image paths and descriptions as a compressed NumPy file.

        Args:
            image_paths (list): List of image paths.
            image_descriptions (list): List of image descriptions (e.g., histograms or VLAD vectors).
            output_file (str): Path to save the NumPy file.
        """
        # Ensure the directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}")
        np.savez(output_file, image_paths=self.image_paths,
                 image_descriptions=self.image_descriptions)
        print(f"Descriptions saved to {output_file}")

    def load_descriptions_from_numpy(self, input_file="image_descriptions.npz"):
        """
        Load image paths and descriptions from a compressed NumPy file.

        Args:
            input_file (str): Path to the NumPy file.

        Returns:
            tuple: A tuple (image_paths, image_descriptions).
        """
        # Check if the file exists
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"File not found: {input_file}")

        data = np.load(input_file, allow_pickle=True)
        self.image_paths = data["image_paths"].tolist()
        self.image_descriptions = data["image_descriptions"]

    def save_vocabulary(self, output_file="vocabulary.pkl"):
        """
        Save the KMeans vocabulary to a file.

        Args:
            kmeans (KMeans): Trained KMeans model.
            output_file (str): Path to save the vocabulary.
        """
        # Ensure the directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}")

        # Save the vocabulary
        with open(output_file, "wb") as f:
            pickle.dump(self.kmeans, f)
        print(f"Vocabulary saved to {output_file}")

    def load_vocabulary(self, input_file="vocabulary.pkl"):
        """
        Load the KMeans vocabulary from a file.

        Args:
            input_file (str): Path to the vocabulary file.

        Returns:
            KMeans: Loaded KMeans model.
        """
        # Check if the file exists
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"File not found: {input_file}")

        # Load the vocabulary
        with open(input_file, "rb") as f:
            self.kmeans = pickle.load(f)
        print(f"Vocabulary loaded from {input_file}")


def main():
    # Example Usage
    directory = "task_5/database"
    # Initialize the pipeline (choose 'BoVW' or 'VLAD')
    pipeline = VPR(method='VLAD', n_clusters=1000)

    # Run the pipeline
    pipeline.run_pipeline(
        directory=directory)

    # Print results
    for i, desc in enumerate(pipeline.image_descriptions):
        print(f"Image: {pipeline.image_paths[i]}")
        print(f"Description: {desc}")


if __name__ == '__main__':
    main()
