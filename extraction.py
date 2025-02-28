import os
import cv2
import numpy as np
import pandas as pd
import logging
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog, local_binary_pattern
from concurrent.futures import ThreadPoolExecutor
from sklearn.feature_selection import SelectKBest, f_classif
import psutil  # For logging memory usage

# **Set Up Logging**
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/feature_extraction.log",
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

def log_and_print(message):
    """Helper function to log and print messages."""
    print(message)
    logging.info(message)

def log_memory_usage():
    """Logs the current memory usage."""
    process = psutil.Process(os.getpid())
    mem_usage = process.memory_info().rss / (1024 ** 2)  # Convert to MB
    log_and_print(f"Memory Usage: {mem_usage:.2f} MB")

log_and_print("Starting Feature Extraction for CK+ Dataset...")

# **Define Paths**
# input_folder = "images/CK+/"
# output_parquet = "Features/ckplus_features.parquet"

input_folder = "images/FER2013/"
output_parquet = "Features/FER_features.parquet"

# **Feature Extraction Parameters**
IMAGE_SIZE = (128, 128)  # Resize images to 128x128  
HOG_PARAMS = {
    "pixels_per_cell": (8, 8),
    "cells_per_block": (2, 2),
    "orientations": 9,
    "feature_vector": True
}
LBP_PARAMS = {
    "P": 8,  # Number of circularly symmetric neighbor set points
    "R": 1,  # Radius of LBP
    "method": "uniform"
}
PCA_VARIANCE_THRESHOLD = 0.95  # Keep 95% variance
SELECT_K = 300  # Select top 300 features

# **Initialize Scaler**
scaler = StandardScaler()

def apply_clahe(image):
    """Applies CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

def extract_lbp_features(image):
    """Extracts LBP features from an image."""
    lbp = local_binary_pattern(image, LBP_PARAMS["P"], LBP_PARAMS["R"], method=LBP_PARAMS["method"])
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, LBP_PARAMS["P"] + 3), density=True)
    return hist

def process_image(img_path, class_name):
    """Loads an image, extracts HOG + LBP features, and returns them."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    img_resized = cv2.resize(img, IMAGE_SIZE)
    img_eq = apply_clahe(img_resized)  # Use CLAHE for contrast enhancement

    # **Extract Features**
    hog_features = hog(img_eq, **HOG_PARAMS)
    lbp_features = extract_lbp_features(img_eq)

    # **Combine Features**
    combined_features = np.hstack((hog_features, lbp_features))
    return combined_features, class_name

# **Process Images Using Multithreading**
features, labels = [], []

with ThreadPoolExecutor() as executor:
    futures = []
    for class_name in os.listdir(input_folder):
        class_path = os.path.join(input_folder, class_name)
        if os.path.isdir(class_path):
            log_and_print(f"Processing class: {class_name}")
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                futures.append(executor.submit(process_image, img_path, class_name))

    for future in futures:
        result = future.result()
        if result:
            features.append(result[0])
            labels.append(result[1])

log_and_print("Feature extraction complete. Converting to NumPy arrays...")
log_memory_usage()

# **Convert to NumPy Array**
features = np.array(features)
labels = np.array(labels)

# **Apply Feature Scaling**
log_and_print("Applying feature scaling...")
features_scaled = scaler.fit_transform(features)
log_memory_usage()

# **Compute PCA Explained Variance**
pca_temp = PCA().fit(features_scaled)
explained_variance = np.cumsum(pca_temp.explained_variance_ratio_)
optimal_components = np.argmax(explained_variance >= PCA_VARIANCE_THRESHOLD) + 1  # **Keep 95% variance**
log_and_print(f"Optimal PCA components for {PCA_VARIANCE_THRESHOLD*100}% variance: {optimal_components}")

# **Apply PCA**
log_and_print(f"Applying PCA with {optimal_components} components...")
pca = PCA(n_components=optimal_components)
X_pca = pca.fit_transform(features_scaled)
log_memory_usage()

# **Apply Feature Selection (SelectKBest)**
log_and_print(f"Applying SelectKBest to select top {SELECT_K} features...")
selector = SelectKBest(f_classif, k=SELECT_K)
X_selected = selector.fit_transform(X_pca, labels)
log_memory_usage()

# **Save Features to Parquet**
log_and_print("Saving extracted features to Parquet file...")
df = pd.DataFrame(X_selected)
df.insert(0, "label", labels)
df.to_parquet(output_parquet, index=False)

log_and_print(f"Feature extraction completed successfully! Saved to {output_parquet}")
