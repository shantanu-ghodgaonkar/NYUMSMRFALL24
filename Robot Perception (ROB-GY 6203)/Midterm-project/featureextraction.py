import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.mixed_precision import set_global_policy
import tensorflow as tf
from joblib import dump
from pathlib import Path
import time
import os

# Set global mixed precision policy
set_global_policy('mixed_float16')

def create_feature_extractor(use_gpu=False):
    """Initialize the VGG16 feature extractor with mixed precision."""
    base_model = VGG16(weights='imagenet', include_top=False, pooling='avg')
    feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)
    
    if use_gpu and tf.config.list_physical_devices('GPU'):
        print("Using GPU for feature extraction")
    
    return feature_extractor

def preprocess_image(path):
    """Load and preprocess a single image."""
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = preprocess_input(image)
    return image

def load_image_paths_as_dataset(image_paths, batch_size):
    """Create a tf.data dataset for batched image loading and preprocessing."""
    path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    image_ds = path_ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    image_ds = image_ds.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    return image_ds

def extract_features_batch(feature_extractor, image_paths, batch_size=32):
    """Extract features from images using efficient tf.data pipelines."""
    features = []
    total_batches = len(image_paths) // batch_size + (1 if len(image_paths) % batch_size else 0)
    
    dataset = load_image_paths_as_dataset(image_paths, batch_size)
    
    print(f"Processing {len(image_paths)} images in {total_batches} batches.")
    start_time = time.time()
    
    for i, batch_images in enumerate(dataset):
        batch_features = feature_extractor.predict(batch_images, verbose=0)
        features.extend(batch_features)
        
        # Print progress with time estimate
        elapsed_time = time.time() - start_time
        avg_time_per_batch = elapsed_time / (i + 1)
        remaining_batches = total_batches - (i + 1)
        estimated_time = remaining_batches * avg_time_per_batch
        
        print(f"Processed batch {i+1}/{total_batches} "
              f"({(i+1)/total_batches*100:.1f}%) - "
              f"Est. time remaining: {estimated_time/60:.1f} minutes")
    
    return np.array(features, dtype=np.float32)

def build_feature_database(directory_path, output_dir, batch_size=32, use_gpu=False):
    """Extract and save features for all images in the directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image paths
    image_paths = [
        str(p) for p in Path(directory_path).rglob("*")
        if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}
    ]
    
    if not image_paths:
        raise ValueError(f"No images found in {directory_path}")
    
    print(f"Found {len(image_paths)} images. Starting feature extraction...")
    
    # Initialize feature extractor
    feature_extractor = create_feature_extractor(use_gpu)
    
    # Extract features
    features = extract_features_batch(feature_extractor, image_paths, batch_size)
    
    # Save features and paths using joblib
    feature_file = os.path.join(output_dir, 'image_features.joblib')
    paths_file = os.path.join(output_dir, 'image_paths.joblib')
    
    print(f"\nSaving features to {feature_file}")
    dump(features, feature_file, compress=3)
    
    print(f"Saving paths to {paths_file}")
    dump(image_paths, paths_file, compress=3)
    
    print(f"\nProcessing complete!")
    print(f"Total images processed: {len(image_paths)}")
    print(f"Feature shape: {features.shape}")
    print(f"Files saved in: {output_dir}")

if __name__ == "__main__":
    # Configuration
    IMAGE_DIR = "C:\\Users\\akank\\vis_nav_player\\data\\Images_1"  # Directory containing your images
    OUTPUT_DIR = "C:\\Users\\akank\\vis_nav_player\\feature.joblib"  # Directory to save the feature files
    BATCH_SIZE = 32                     # Adjust based on your GPU memory
    USE_GPU = True                      # Set to False if you want to use CPU only
    
    try:
        build_feature_database(
            directory_path=IMAGE_DIR,
            output_dir=OUTPUT_DIR,
            batch_size=BATCH_SIZE,
            use_gpu=USE_GPU
        )
    except Exception as e:
        print(f"Error during processing: {str(e)}")
