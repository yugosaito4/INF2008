import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.base import clone
import logging
import collections

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

### **1️⃣ Load and Normalize HOG + LBP Features from Separate Files**
logger.info("Loading HOG and LBP features from separate files...")

hog_data = pd.read_csv("HOG_Extraction/hog_facial_expression_features_192.csv")  # HOG features
lbp_data = pd.read_csv("LBP_Extraction/lbp_features_all_emotions.csv")  # LBP features

assert hog_data.shape[0] == lbp_data.shape[0], "Mismatch in number of samples between HOG and LBP"

hog_features = hog_data.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').values
lbp_features = lbp_data.iloc[:, :-1].apply(pd.to_numeric, errors='coerce').values

# Normalize using MinMaxScaler
hog_scaler = MinMaxScaler()
lbp_scaler = MinMaxScaler()

hog_features_scaled = hog_scaler.fit_transform(hog_features)
lbp_features_scaled = lbp_scaler.fit_transform(lbp_features)

# Combine features
X_combined = np.hstack((hog_features_scaled, lbp_features_scaled))

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(lbp_data.iloc[:, -1])

# Check class distribution
class_distribution = collections.Counter(y)
logger.info(f"Class distribution before SMOTE: {class_distribution}")

# Apply SMOTE to balance classes
smote = SMOTE(random_state=42)
X_combined, y = smote.fit_resample(X_combined, y)

logger.info(f"Class distribution after SMOTE: {collections.Counter(y)}")
logger.info(f"Dataset loaded: {X_combined.shape[0]} samples, {X_combined.shape[1]} combined features.")

### **2️⃣ Split Dataset: 80% Train, 20% Test**
X_train_full, X_test, y_train_full, y_test = train_test_split(X_combined, y, test_size=0.2, shuffle=True, random_state=42, stratify=y)

### **3️⃣ Perform 5-Fold Stratified Cross-Validation on Training Set**
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define Random Forest model
model = RandomForestClassifier(
    bootstrap=True,
    n_estimators=300,
    max_depth=25,
    min_samples_split=100,
    min_samples_leaf=30,
    max_features='sqrt',
    class_weight=None,
    random_state=42,
    n_jobs=-1
)

logger.info("Performing 5-Fold Stratified Cross-Validation on Training Set...")

best_model = None
best_score = 0
best_train_idx, best_val_idx = None, None

for train_idx, val_idx in skf.split(X_train_full, y_train_full):
    # Split the training set further into training and validation
    X_train_fold, X_val_fold = X_train_full[train_idx], X_train_full[val_idx]
    y_train_fold, y_val_fold = y_train_full[train_idx], y_train_full[val_idx]

    # Train model on fold
    model_fold = clone(model)  # Clone ensures a fresh model each time
    model_fold.fit(X_train_fold, y_train_fold)
    
    # Evaluate on validation set
    val_acc = model_fold.score(X_val_fold, y_val_fold)
    logger.info(f"Fold validation accuracy: {val_acc:.5f}")

    # Keep track of the best model
    if val_acc > best_score:
        best_score = val_acc
        best_model = clone(model_fold)
        best_train_idx, best_val_idx = train_idx, val_idx

logger.info(f"Best fold validation accuracy: {best_score:.5f}")

# Train on best fold
X_train_best, X_val_best = X_train_full[best_train_idx], X_train_full[best_val_idx]
y_train_best, y_val_best = y_train_full[best_train_idx], y_train_full[best_val_idx]

best_model.fit(X_train_best, y_train_best)

### **4️⃣ Final Testing on Separate 20% Test Set**
logger.info("Testing on Final 20% Test Set...")

# Predictions
y_pred = best_model.predict(X_test)
y_pred_labels = label_encoder.inverse_transform(y_pred)
y_test_labels = label_encoder.inverse_transform(y_test)

# Classification Report & Confusion Matrix
classification = classification_report(y_test_labels, y_pred_labels)
logger.info("Classification Report:\n" + classification)

conf_matrix = confusion_matrix(y_test_labels, y_pred_labels)
logger.info("Confusion Matrix:\n" + str(conf_matrix))

# Accuracy
train_acc = best_model.score(X_train_full, y_train_full)  # Accuracy on entire training data
test_acc = best_model.score(X_test, y_test)  # Accuracy on final test data
logger.info(f"Training Accuracy: {train_acc:.5f}")
logger.info(f"Test Accuracy: {test_acc:.5f}")
