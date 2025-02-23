import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

### **1️⃣ Load and Normalize HOG + LBP Features**
logger.info("Loading HOG and LBP features...")

hog_data = pd.read_csv("HOG_Extraction/hog_facial_expression_features_192.csv")
lbp_data = pd.read_csv("LBP_Extraction/lbp_features_all_emotions.csv")

assert hog_data.shape[0] == lbp_data.shape[0], "Mismatch in number of samples between HOG and LBP"

hog_features = hog_data.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").values
lbp_features = lbp_data.iloc[:, :-1].apply(pd.to_numeric, errors="coerce").values
X_combined = np.hstack((hog_features, lbp_features))

# Encode labels
label_encoder = pd.factorize(lbp_data.iloc[:, -1])
y = label_encoder[0]
emotion_labels = label_encoder[1]

logger.info(f"Dataset loaded: {X_combined.shape[0]} samples, {X_combined.shape[1]} merged features.")

### **2️⃣ Apply SMOTE for Class Balancing**
smote = SMOTE(random_state=42)
X_combined, y = smote.fit_resample(X_combined, y)

### **3️⃣ Split Dataset: 80% Train, 20% Test**
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_combined, y, test_size=0.2, shuffle=True, random_state=42, stratify=y
)

### **4️⃣ Apply MinMaxScaler**
logger.info("Applying MinMaxScaler...")
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_full)
X_test_scaled = scaler.transform(X_test)

### **5️⃣ Feature Selection Using Random Forest**
logger.info("Performing Feature Selection Using Random Forest...")

feature_selector = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
feature_selector.fit(X_train_scaled, y_train_full)

importances = feature_selector.feature_importances_
top_k_percent = 25  # Keep top 25% most important features
threshold = np.percentile(importances, 100 - top_k_percent)
selected_features = np.where(importances >= threshold)[0]

X_train_selected = X_train_scaled[:, selected_features]
X_test_selected = X_test_scaled[:, selected_features]

logger.info(f"Selected {len(selected_features)} important features out of {X_combined.shape[1]}.")

### **6️⃣ Hyperparameter Tuning Using GridSearchCV**
logger.info("Optimizing Random Forest Model with GridSearchCV...")

param_grid = {
    "n_estimators": [100, 200, 300, 500, 700, 1000],  # Include original 300, add wider range
    "max_depth": [15, 20, 25, 30, None],  # Include original 25, allow unlimited depth
    "min_samples_split": [10, 20, 50, 100],  # Include original 100
    "min_samples_leaf": [5, 10, 20, 30],  # Include original 30
    "max_features": ["sqrt", "log2", None],  # Try different feature selections
    "class_weight": ["balanced", None]  # Include original None
}


rf = RandomForestClassifier(random_state=42, n_jobs=-1)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=1)
grid_search.fit(X_train_selected, y_train_full)

best_model = grid_search.best_estimator_
logger.info(f"Best parameters: {grid_search.best_params_}")
logger.info(f"Best cross-validation score: {grid_search.best_score_:.3f}")

### **7️⃣ Evaluate on Test Set**
logger.info("Testing on Final 20% Test Set...")

y_pred = best_model.predict(X_test_selected)

y_pred_labels = [emotion_labels[i] for i in y_pred]
y_test_labels = [emotion_labels[i] for i in y_test]

classification = classification_report(y_test_labels, y_pred_labels, target_names=emotion_labels)
conf_matrix = confusion_matrix(y_test_labels, y_pred_labels)

train_acc = best_model.score(X_train_selected, y_train_full)
test_acc = best_model.score(X_test_selected, y_test)

logger.info(f"Training Accuracy: {train_acc:.5f}")
logger.info(f"Test Accuracy: {test_acc:.5f}")
