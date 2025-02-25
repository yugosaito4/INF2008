import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import logging

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Load Data
logger.info("Loading HOG and LBP features...")
hog_data = pd.read_csv("HOG_Extraction/hog_facial_expression_features_192.csv")
lbp_data = pd.read_csv("LBP_Extraction/lbp_features_all_emotions.csv")

hog_features = hog_data.iloc[:, 1:].values
lbp_features = lbp_data.iloc[:, :-1].values
X_combined = np.hstack((hog_features, lbp_features))

# Encode Labels
y, emotion_labels = lbp_data.iloc[:, -1].factorize()

# Split Data (No SMOTE)
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.2, random_state=42, stratify=y
)

# Feature Selection using RandomForest (No Scaling)
logger.info("Performing Feature Selection Using Random Forest...")
feature_selector = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
feature_selector.fit(X_train, y_train)

importances = feature_selector.feature_importances_
top_k_percent = 25  # Keep top 25% most important features
threshold = np.percentile(importances, 100 - top_k_percent)
selected_features = np.where(importances >= threshold)[0]

X_train_selected = X_train[:, selected_features]
X_test_selected = X_test[:, selected_features]

logger.info(f"Selected {len(selected_features)} important features out of {X_combined.shape[1]}.")

# Train Final Model
logger.info("Training Final Random Forest Model...")
best_model = RandomForestClassifier(n_estimators=500, max_depth=30, min_samples_split=10, min_samples_leaf=1,
                                    class_weight="balanced", random_state=42, n_jobs=-1)
best_model.fit(X_train_selected, y_train)

# Evaluate
y_pred = best_model.predict(X_test_selected)

y_pred_labels = [emotion_labels[i] for i in y_pred]
y_test_labels = [emotion_labels[i] for i in y_test]

classification = classification_report(y_test_labels, y_pred_labels, target_names=emotion_labels)
conf_matrix = confusion_matrix(y_test_labels, y_pred_labels)

logger.info("Classification Report:\n" + classification)
logger.info("Confusion Matrix:\n" + str(conf_matrix))

train_acc = best_model.score(X_train_selected, y_train)
test_acc = best_model.score(X_test_selected, y_test)
logger.info(f"Training Accuracy: {train_acc:.5f}")
logger.info(f"Test Accuracy: {test_acc:.5f}")
