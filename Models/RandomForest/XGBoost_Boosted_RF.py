import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import logging

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# ✅ Load the optimized dataset
logger.info("Loading Optimized Features Dataset...")
optimized_data = pd.read_parquet("C:/Users/Admin/Downloads/optimized_features.parquet")

# ✅ Extract features and labels
X_combined = optimized_data.iloc[:, 1:].values  # All columns except label
y, emotion_labels = optimized_data["label"].factorize()  # Encode emotion labels

logger.info(f"Emotion Labels Mapping: {dict(enumerate(emotion_labels))}")

# ✅ Split dataset (80% train, 20% test)
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_combined, y, test_size=0.2, random_state=42, stratify=y
)

# ✅ Train XGBoost Boosted Random Forest
best_model = xgb.XGBClassifier(
    booster="dart",  # Use dropout additive regression trees (boosted RF)
    n_estimators=1000,  # Best number of trees
    max_depth=20,  # Maximum tree depth
    learning_rate=0.05,  # Control step size of updates
    subsample=0.8,  # Subsample ratio (bagging)
    colsample_bytree=0.8,  # Feature subsampling
    min_child_weight=1,  # Minimum samples in leaf
    random_state=42,
    n_jobs=-1
)

logger.info("Training the final XGBoost Boosted Random Forest model...")
best_model.fit(X_train_full, y_train_full)

# ✅ Evaluate on test set
logger.info("Evaluating on test set...")
y_pred = best_model.predict(X_test)

# ✅ Convert index-based labels back to emotion names
y_pred_labels = [emotion_labels[i] for i in y_pred]
y_test_labels = [emotion_labels[i] for i in y_test]

# ✅ Classification Report & Confusion Matrix (With Emotion Labels)
classification = classification_report(y_test_labels, y_pred_labels, target_names=emotion_labels)
conf_matrix = confusion_matrix(y_test_labels, y_pred_labels)

logger.info("Classification Report:\n" + classification)
logger.info("Confusion Matrix:\n" + str(conf_matrix))

# ✅ Compute Accuracy
train_acc = best_model.score(X_train_full, y_train_full)
test_acc = best_model.score(X_test, y_test)
logger.info(f"Training Accuracy: {train_acc:.5f}")
logger.info(f"Test Accuracy: {test_acc:.5f}")
