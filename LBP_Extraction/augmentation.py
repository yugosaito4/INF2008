from keras.preprocessing.image import ImageDataGenerator
import os
import cv2
import numpy as np


#augmentation of disgust images
disgust_path = "C:\\Users\\PC\\OneDrive - Singapore Institute Of Technology\\Desktop\\Work\\School\\INF2008\\images\\train\\disgust"
augmented_path = "C:\\Users\\PC\\OneDrive - Singapore Institute Of Technology\\Desktop\\Work\\School\\INF2008\\images\\train\\disgust"

# Image augmentation setup
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Load and augment disgust images
for img_name in os.listdir(disgust_path):
    img_path = os.path.join(disgust_path, img_name)
    img = cv2.imread(img_path)
    img = np.expand_dims(img, axis=0)  # Expand dimensions for augmentation

    # Save augmented images
    i = 0
    for batch in datagen.flow(img, batch_size=1, save_to_dir=augmented_path, save_prefix='aug', save_format='jpg'):
        i += 1
        if i > 5:  # Generate 5 augmented images per original image
            break
