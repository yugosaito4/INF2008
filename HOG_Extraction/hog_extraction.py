import os
import numpy as np
import pandas as pd
import cv2
from skimage import io, color, feature
from skimage.transform import resize
import matplotlib.pyplot as plt

# Load OpenCV's Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_image(image_path, visualize=False):
    # Load image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    if len(faces) > 0:
        x, y, w, h = faces[0]
        face_image = gray[y:y+h, x:x+w]
    else:
        face_image = gray

    face_image = resize(face_image, (64, 64), anti_aliasing=True)

    # # Extract HOG features
    # if visualize:
    #     hog_features, hog_image = feature.hog(face_image, orientations=8, pixels_per_cell=(32, 32),
    #                                           cells_per_block=(1, 1), visualize=True)
    #     # Display the visualization
    #     plt.figure(figsize=(8, 4))
    #     plt.subplot(121)
    #     plt.imshow(face_image, cmap='gray')
    #     plt.title('Processed Face')
    #     plt.axis('off')

    #     plt.subplot(122)
    #     plt.imshow(hog_image, cmap='gray')
    #     plt.title('HOG Visualization')
    #     plt.axis('off')

    #     plt.show(block=False)
    #     plt.pause(0.5)
    #     plt.close()
    # else:
    #     hog_features = feature.hog(face_image, orientations=8, pixels_per_cell=(32, 32),
    #                                cells_per_block=(1, 1), visualize=False)
        
    hog_features = feature.hog(face_image, orientations=8, pixels_per_cell=(32, 32),
                                   cells_per_block=(1, 1), visualize=False)

    return hog_features

def extract_features_labels(directory, visualize=False):
    """Process all images in the directory and extract features and labels."""
    data = []
    labels = []

    # Iterate over each folder
    for expression in os.listdir(directory):
        folder_path = os.path.join(directory, expression)
        
        if os.path.isdir(folder_path):
            # Process each image in the folder
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                
                try:
                    # Preprocess the image and extract HOG features
                    hog_features = preprocess_image(file_path, visualize)
                    data.append(hog_features)
                    labels.append(expression)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    return labels, np.array(data)

# Adjust the path to the directory containing your training images
labels, features = extract_features_labels('../images/train', visualize=True)

# Create DataFrame with labels first
df = pd.DataFrame(features)
df.insert(0, 'label', labels)

# Rename columns to 'label', 'bin_1', 'bin_2', ..., 'bin_n'
columns = ['label'] + [f'bin_{i+1}' for i in range(df.shape[1] - 1)]
df.columns = columns

# Save DataFrame to CSV
df.to_csv('hog_facial_expression_features.csv', index=False)

print("Features saved to 'hog_features_all_emotions.csv'")