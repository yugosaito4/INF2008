import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.base import clone
import logging

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Load the optimized dataset
logger.info("Loading Controlled Features Dataset...")
optimized_data = pd.read_parquet("Features/ckplus_features.parquet")

# Extract features and labels
X_combined = optimized_data.iloc[:, 1:].values  # All columns except label
y, emotion_labels = optimized_data["label"].factorize()  # Encode emotion labels

logger.info(f"Emotion Labels Mapping: {dict(enumerate(emotion_labels))}")

# Split dataset (80% train, 20% test)
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_combined, y, test_size=0.2, random_state=42, stratify=y
)

# Set up 5-Fold Stratified Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define the best model (Best params dervied from previously performing GridSearch)
best_model = RandomForestClassifier(
    n_estimators=1000,       # Best number of trees
    max_depth=20,            # Maximum tree depth
    min_samples_split=5,     # Minimum samples to split
    min_samples_leaf=1,      # Minimum samples per leaf
    class_weight="balanced", # Handle class imbalance
    random_state=42, 
    n_jobs=-1
)

best_cv_model = None
best_cv_score = 0

logger.info("Performing 5-Fold Stratified Cross-Validation on Training Set...")

# Iterate over each fold
for train_idx, val_idx in skf.split(X_train_full, y_train_full):
    # Create train-validation split for each fold
    X_train_fold, X_val_fold = X_train_full[train_idx], X_train_full[val_idx]
    y_train_fold, y_val_fold = y_train_full[train_idx], y_train_full[val_idx]

    # Clone and train model on fold
    model_fold = clone(best_model)
    model_fold.fit(X_train_fold, y_train_fold)

    # Evaluate fold performance
    val_acc = model_fold.score(X_val_fold, y_val_fold)
    logger.info(f"Fold validation accuracy: {val_acc:.5f}")

    # Keep track of the best-performing model
    if val_acc > best_cv_score:
        best_cv_score = val_acc
        best_cv_model = model_fold  # Store the trained model

logger.info(f"Best cross-validation accuracy: {best_cv_score:.5f}")

# Train the best model on the entire training set
logger.info("Training the final Random Forest model on full training set...")
best_cv_model.fit(X_train_full, y_train_full)

# Evaluate on test set
logger.info("Evaluating on test set...")
y_pred = best_cv_model.predict(X_test)

# Convert index-based labels back to emotion names
y_pred_labels = [emotion_labels[i] for i in y_pred]
y_test_labels = [emotion_labels[i] for i in y_test]

# Classification Report & Confusion Matrix (With Emotion Labels)
classification = classification_report(y_test_labels, y_pred_labels, target_names=emotion_labels)
conf_matrix = confusion_matrix(y_test_labels, y_pred_labels)

logger.info("Classification Report:\n" + classification)
logger.info("Confusion Matrix:\n" + str(conf_matrix))

# Compute Accuracy
train_acc = best_cv_model.score(X_train_full, y_train_full)
test_acc = best_cv_model.score(X_test, y_test)
logger.info(f"Training Accuracy: {train_acc:.5f}")
logger.info(f"Test Accuracy: {test_acc:.5f}")
