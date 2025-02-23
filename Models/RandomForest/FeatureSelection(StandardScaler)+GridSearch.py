import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

logger.info("Loading HOG and LBP features...")

hog_data = pd.read_csv("HOG_Extraction/hog_facial_expression_features_192.csv")
lbp_data = pd.read_csv("LBP_Extraction/lbp_features_all_emotions.csv")

assert hog_data.shape[0] == lbp_data.shape[0], "Mismatch in number of samples between HOG and LBP"

hog_features = hog_data.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").values
lbp_features = lbp_data.iloc[:, :-1].apply(pd.to_numeric, errors="coerce").values
X_combined = np.hstack((hog_features, lbp_features))

label_encoder = pd.factorize(lbp_data.iloc[:, -1])
y = label_encoder[0]
emotion_labels = label_encoder[1]

logger.info(f"Dataset loaded: {X_combined.shape[0]} samples, {X_combined.shape[1]} merged features.")

smote = SMOTE(random_state=42)
X_combined, y = smote.fit_resample(X_combined, y)

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_combined, y, test_size=0.2, shuffle=True, random_state=42, stratify=y
)

### **Apply StandardScaler**
logger.info("Applying StandardScaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_full)
X_test_scaled = scaler.transform(X_test)

logger.info("Performing Feature Selection Using Random Forest...")

feature_selector = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
feature_selector.fit(X_train_scaled, y_train_full)

importances = feature_selector.feature_importances_
threshold = np.percentile(importances, 75)  # Keep top 25%
selected_features = np.where(importances >= threshold)[0]

X_train_selected = X_train_scaled[:, selected_features]
X_test_selected = X_test_scaled[:, selected_features]

logger.info("Optimizing Random Forest Model with GridSearchCV...")

param_grid = {
    "n_estimators": [300, 500, 700],
    "max_depth": [15, 20, 25],
    "min_samples_split": [10, 20],
    "min_samples_leaf": [5, 10],
    "class_weight": ["balanced"]
}

rf = RandomForestClassifier(random_state=42, n_jobs=-1)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=1)
grid_search.fit(X_train_selected, y_train_full)

best_model = grid_search.best_estimator_

logger.info("Testing on Final 20% Test Set...")

y_pred = best_model.predict(X_test_selected)

classification = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

train_acc = best_model.score(X_train_selected, y_train_full)
test_acc = best_model.score(X_test_selected, y_test)

logger.info(f"Training Accuracy: {train_acc:.5f}")
logger.info(f"Test Accuracy: {test_acc:.5f}")
