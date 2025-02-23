import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.base import clone
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

# Encode labels
y = lbp_data.iloc[:, -1].factorize()[0]

# Apply SMOTE
smote = SMOTE(random_state=42)
X_combined, y = smote.fit_resample(X_combined, y)

# Split dataset
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_combined, y, test_size=0.2, random_state=42, stratify=y
)

# 5-Fold Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = RandomForestClassifier(n_estimators=300, max_depth=25, random_state=42, n_jobs=-1)

best_model = None
best_score = 0

logger.info("Performing 5-Fold Stratified Cross-Validation...")

for train_idx, val_idx in skf.split(X_train_full, y_train_full):
    X_train_fold, X_val_fold = X_train_full[train_idx], X_train_full[val_idx]
    y_train_fold, y_val_fold = y_train_full[train_idx], y_train_full[val_idx]

    model_fold = clone(model)
    model_fold.fit(X_train_fold, y_train_fold)  # Train the model

    val_acc = model_fold.score(X_val_fold, y_val_fold)
    logger.info(f"Fold validation accuracy: {val_acc:.5f}")

    if val_acc > best_score:
        best_score = val_acc
        best_model = model_fold  # ✅ Store the trained model

logger.info(f"Best fold validation accuracy: {best_score:.5f}")

# ✅ Ensure the best model is fitted before testing
if best_model is None:
    logger.error("No best model was found. Exiting...")
    exit()

# Final testing
y_pred = best_model.predict(X_test)
logger.info("Classification Report:\n" + classification_report(y_test, y_pred))
logger.info("Confusion Matrix:\n" + str(confusion_matrix(y_test, y_pred)))

train_acc = best_model.score(X_train_full, y_train_full)
test_acc = best_model.score(X_test, y_test)
logger.info(f"Training Accuracy: {train_acc:.5f}")
logger.info(f"Test Accuracy: {test_acc:.5f}")
