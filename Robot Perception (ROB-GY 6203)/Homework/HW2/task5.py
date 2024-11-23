import os
import cv2
import numpy as np
import torch
import pickle
from sklearn.cluster import MiniBatchKMeans
from torch.nn.functional import normalize
import matplotlib.pyplot as plt
from tqdm import tqdm


class ImageRetrieval:
    def __init__(self, query_dir, database_dir, output_dir, num_clusters=64, device=None):
        self.query_dir = query_dir
        self.database_dir = database_dir
        self.output_dir = output_dir
        self.num_clusters = num_clusters
        self.device = device if device else (
            "cuda" if torch.cuda.is_available() else "cpu")

        # Define file paths for saving outputs
        self.database_descriptors_file = os.path.join(
            output_dir, "database_descriptors.pkl")
        self.kmeans_centroids_file = os.path.join(
            output_dir, "kmeans_centroids.pkl")
        self.database_vlads_file = os.path.join(
            output_dir, "database_vlads.pkl")

        # Collect images
        self.query_images = [
            os.path.join(self.query_dir, f)
            for f in os.listdir(self.query_dir)
            if f.lower().endswith(('png', 'jpg', 'jpeg'))
        ]
        self.database_images = [
            os.path.join(self.database_dir, f)
            for f in os.listdir(self.database_dir)
            if f.lower().endswith(('png', 'jpg', 'jpeg'))
        ]

    def extract_sift_features(self, image_path):
        """Extract SIFT features from an image."""
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        sift = cv2.SIFT_create()
        _, descriptors = sift.detectAndCompute(img, None)
        return descriptors

    def build_vlad(self, descriptors, kmeans_centroids):
        """Compute VLAD encoding for the descriptors."""
        num_clusters = kmeans_centroids.shape[0]
        descriptors = torch.tensor(descriptors, device=self.device).float()
        centroids = torch.tensor(kmeans_centroids, device=self.device).float()

        labels = torch.argmin(torch.cdist(descriptors, centroids), dim=1)
        vlad = torch.zeros(
            (num_clusters, descriptors.size(1)), device=self.device)

        for i in range(num_clusters):
            if torch.sum(labels == i) > 0:
                cluster_descriptors = descriptors[labels == i, :]
                cluster_center = centroids[i]
                vlad[i] = torch.sum(cluster_descriptors -
                                    cluster_center, dim=0)

        vlad = vlad.flatten()
        vlad = normalize(vlad, p=2, dim=0)  # L2 normalization
        return vlad.cpu().numpy()

    def find_matches(self, query_vlad, database_vlads, threshold=0.7):
        """Find matches for a query VLAD vector using cosine similarity."""
        query_vlad = torch.tensor(query_vlad, device=self.device).float()
        database_vlads = torch.tensor(
            database_vlads, device=self.device).float()

        similarities = torch.nn.functional.cosine_similarity(
            query_vlad.unsqueeze(0), database_vlads)
        matches = [
            (sim.item(), self.database_images[i])
            for i, sim in enumerate(similarities)
            if sim > threshold
        ]
        matches.sort(reverse=True, key=lambda x: x[0])  # Sort by similarity
        return matches

    def load_or_compute_descriptors(self, image_paths, descriptor_file):
        if os.path.exists(descriptor_file):
            print(f"Loading descriptors from {descriptor_file}...")
            with open(descriptor_file, "rb") as f:
                return pickle.load(f)
        else:
            print("Extracting SIFT features for images...")
            descriptors = [
                self.extract_sift_features(img) for img in tqdm(image_paths, desc="Extracting SIFT")
            ]
            with open(descriptor_file, "wb") as f:
                pickle.dump(descriptors, f)
            return descriptors

    def load_or_compute_kmeans(self, descriptors, kmeans_file):
        if os.path.exists(kmeans_file):
            print(f"Loading k-means centroids from {kmeans_file}...")
            with open(kmeans_file, "rb") as f:
                return pickle.load(f)
        else:
            print("Performing k-means clustering...")
            kmeans = MiniBatchKMeans(
                n_clusters=self.num_clusters, random_state=42, batch_size=500
            )
            kmeans.fit(descriptors)
            with open(kmeans_file, "wb") as f:
                pickle.dump(kmeans.cluster_centers_, f)
            return kmeans.cluster_centers_

    def load_or_compute_vlads(self, descriptors, centroids, vlad_file):
        if os.path.exists(vlad_file):
            print(f"Loading VLAD descriptors from {vlad_file}...")
            with open(vlad_file, "rb") as f:
                return pickle.load(f)
        else:
            print("Computing VLAD descriptors for images...")
            vlads = [
                self.build_vlad(desc, centroids)
                for desc in tqdm(descriptors, desc="Computing VLADs")
                if desc is not None
            ]
            with open(vlad_file, "wb") as f:
                pickle.dump(vlads, f)
            return vlads

    def run(self):
        # Step 1: Load or compute descriptors for database images
        database_descriptors = self.load_or_compute_descriptors(
            self.database_images, self.database_descriptors_file
        )

        # Step 2: Perform k-means clustering
        all_descriptors = np.vstack(
            [desc for desc in database_descriptors if desc is not None]
        )
        kmeans_centroids = self.load_or_compute_kmeans(
            all_descriptors, self.kmeans_centroids_file
        )

        # Step 3: Load or compute VLAD descriptors for database images
        database_vlads = self.load_or_compute_vlads(
            database_descriptors, kmeans_centroids, self.database_vlads_file
        )

        # Step 4: Process query images and match
        print("Processing query images...")
        query_descriptors = [
            self.extract_sift_features(img)
            for img in tqdm(self.query_images, desc="Extracting Query SIFT")
        ]
        query_vlads = [
            self.build_vlad(desc, kmeans_centroids)
            for desc in tqdm(query_descriptors, desc="Computing Query VLADs")
            if desc is not None
        ]

        print("Matching query images with database images...")
        results = {}
        for i, q_vlad in enumerate(tqdm(query_vlads, desc="Matching Images")):
            matches = self.find_matches(q_vlad, database_vlads)
            results[self.query_images[i]] = matches

        # Step 5: Display results
        for query, matches in results.items():
            print(f"\nResults for {query}:")
            for similarity, match in matches:
                print(f"Matched with {match}, Similarity: {similarity:.2f}")
                img = cv2.imread(match)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                plt.imshow(img_rgb)
                plt.title(f"Match with Similarity: {similarity:.2f}")
                plt.axis('off')
                plt.show()


if __name__ == "__main__":
    query_dir = "task_5/query/"
    database_dir = "task_5/database/"
    output_dir = "task_5/"

    image_retrieval = ImageRetrieval(query_dir, database_dir, output_dir)
    image_retrieval.run()
