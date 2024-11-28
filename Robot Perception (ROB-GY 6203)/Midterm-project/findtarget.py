import numpy as np
from PIL import Image
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import euclidean_distances
from joblib import load
import re

def extract_features(image_path):
    """Extract features from a single image using VGG16."""
    # Load and prepare the model
    base_model = VGG16(weights='imagenet', include_top=False, pooling='avg')
    feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)
    
    # Process the image
    image = Image.open(image_path).convert('RGB')
    image = image.resize((224, 224))
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)
    
    # Extract features
    features = feature_extractor.predict(image_array, verbose=0)
    return features.flatten().astype(np.float32)

def find_closest_match(target_features, stored_features):
    """Find the closest matching image number based on feature similarity."""
    # Compute distances between target and all stored features
    distances = euclidean_distances([target_features], stored_features)[0]
    
    # Get the index of the closest match
    closest_match = np.argmin(distances)
    return closest_match

def extract_image_number(path):
    """Extract image number from path like 'image_404'"""
    # Find number after 'image_' using regex
    match = re.search(r'image_(\d+)', path)
    if match:
        return int(match.group(1))
    return None

def main():
    # Configuration
    TARGET_IMAGE = "C:\\Users\\akank\\vis_nav_player\\target2.jpg"
    FEATURES_PATH = "C:\\Users\\akank\\vis_nav_player\\feature.joblib\\image_features.joblib"
    PATHS_PATH = "C:\\Users\\akank\\vis_nav_player\\feature.joblib\\image_paths.joblib"
    
    try:
        # Load stored features from joblib files
        stored_features = load(FEATURES_PATH)
        image_paths = load(PATHS_PATH)
        
        # Extract features from target image
        target_features = extract_features(TARGET_IMAGE)
        
        # Find closest match
        closest_index = find_closest_match(target_features, stored_features)
        
        # Get the image number from the matching path
        matching_path = image_paths[closest_index]
        image_number = extract_image_number(matching_path)
        
        # Print just the image number
        if image_number is not None:
            print(image_number)
        else:
            print(f"Error: Could not extract image number from path: {matching_path}")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()