import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import logging

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Load data
logger.info("Loading HOG and LBP features...")
hog_data = pd.read_csv("HOG_Extraction/hog_facial_expression_features_192.csv")
lbp_data = pd.read_csv("LBP_Extraction/lbp_features_all_emotions.csv")

hog_features = hog_data.iloc[:, 1:].values
lbp_features = lbp_data.iloc[:, :-1].values
X_combined = np.hstack((hog_features, lbp_features))

# Encode labels (Save label names)
y, emotion_labels = lbp_data.iloc[:, -1].factorize()

logger.info(f"Emotion Labels Mapping: {dict(enumerate(emotion_labels))}")

# Split dataset (No SMOTE applied)
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_combined, y, test_size=0.2, random_state=42, stratify=y
)

# ✅ Directly using the best parameters found via GridSearchCV
best_model = RandomForestClassifier(
    n_estimators=500,       # Best number of trees
    max_depth=30,           # Maximum tree depth
    min_samples_split=10,   # Minimum samples to split
    min_samples_leaf=1,     # Minimum samples per leaf
    class_weight="balanced",# Handle class imbalance
    random_state=42, 
    n_jobs=-1
)

# Train the best model on the full training set
logger.info("Training the final Random Forest model...")
best_model.fit(X_train_full, y_train_full)

# Final testing on test set
logger.info("Evaluating on test set...")
y_pred = best_model.predict(X_test)

# Convert index-based labels back to emotion names
y_pred_labels = [emotion_labels[i] for i in y_pred]
y_test_labels = [emotion_labels[i] for i in y_test]

# Classification Report & Confusion Matrix (With Emotion Labels)
classification = classification_report(y_test_labels, y_pred_labels, target_names=emotion_labels)
conf_matrix = confusion_matrix(y_test_labels, y_pred_labels)

logger.info("Classification Report:\n" + classification)
logger.info("Confusion Matrix:\n" + str(conf_matrix))

# Compute Accuracy
train_acc = best_model.score(X_train_full, y_train_full)
test_acc = best_model.score(X_test, y_test)
logger.info(f"Training Accuracy: {train_acc:.5f}")
logger.info(f"Test Accuracy: {test_acc:.5f}")
