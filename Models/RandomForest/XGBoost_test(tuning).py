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
optimized_data = pd.read_parquet("C:/Users/Admin/Downloads/Telegram Desktop/ckplus_features.parquet")

# ✅ Extract features and labels
X_combined = optimized_data.iloc[:, 1:].values  # All columns except label
y, emotion_labels = optimized_data["label"].factorize()  # Encode emotion labels

logger.info(f"Emotion Labels Mapping: {dict(enumerate(emotion_labels))}")

# ✅ Split dataset (80% train, 20% test)
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_combined, y, test_size=0.2, random_state=42, stratify=y
)

# ✅ Define XGBoost Standalone Random Forest Model
best_model = xgb.XGBClassifier(
    booster="gbtree",  # ✅ Use tree-based method, not boosting
    num_parallel_tree=1000,  # ✅ Equivalent to n_estimators in RF
    max_depth=20,  # Maximum tree depth
    learning_rate=1,  # ✅ Learning rate must be 1 for Random Forest
    subsample=0.8,  # Row sampling (bagging)
    colsample_bynode=0.8,  # ✅ Feature sampling per split
    min_child_weight=1,  # Minimum samples per leaf
    random_state=42,
    tree_method="hist",  # ✅ Corrected tree method
    device="cuda"  # ✅ Run on GPU
)

logger.info("Training the XGBoost Standalone Random Forest model...")
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
