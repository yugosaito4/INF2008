from skimage.feature import local_binary_pattern
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import cv2



dataPath = "C:\\Users\\PC\\OneDrive - Singapore Institute Of Technology\\Desktop\\Work\\School\\INF2008\\images\\train\\"

radius = 1
n_points = 8 * radius
emotions = ["angry", "happy", "sad", "neutral","fear","disgust"]  
data = []

#extract the features
def extract_lbp_features(image, radius, n_points):
    lbp = local_binary_pattern(image, n_points, radius, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(),
                             bins=np.arange(0, n_points + 3),
                             range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)  # Normalize the histogram
    return hist


for emotion in emotions:
    folder_path = os.path.join(dataPath, emotion)
    print(f"Processing emotion: {emotion}")
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        # Load the image in grayscale
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        assert image.shape == (48, 48), f"Unexpected image size: {image.shape}"
        
        # Extract LBP features
        features = extract_lbp_features(image, radius, n_points)
        
        # Append features and label to the dataset
        data.append(list(features) + [emotion])

# Save all features and labels to a CSV
columns = [f"bin_{i}" for i in range(n_points + 2)] + ["label"]
df = pd.DataFrame(data, columns=columns)
output_csv = "lbp_features_all_emotions.csv"
df.to_csv(output_csv, index=False)
print(f"LBP features extracted and saved to {output_csv}")