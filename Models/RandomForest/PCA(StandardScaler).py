import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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

# Split Dataset (No SMOTE)
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_combined, y, test_size=0.2, random_state=42, stratify=y
)

# Apply StandardScaler (Only on Training Set)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_full)
X_test_scaled = scaler.transform(X_test)  # Use the same transformation

# Apply PCA
pca = PCA(n_components=0.95)  # Keep 95% variance
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

logger.info(f"PCA reduced feature count from {X_combined.shape[1]} to {X_train_pca.shape[1]}.")

# ✅ Use the Best Parameters from GridSearchCV
best_model = RandomForestClassifier(
    n_estimators=500,       # Best number of trees
    max_depth=30,           # Maximum tree depth
    min_samples_split=10,   # Minimum samples to split
    min_samples_leaf=1,     # Minimum samples per leaf
    class_weight="balanced",# Handle class imbalance
    random_state=42, 
    n_jobs=-1
)

# 5-Fold Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_score = 0

logger.info("Performing 5-Fold Stratified Cross-Validation...")

for train_idx, val_idx in skf.split(X_train_pca, y_train_full):
    X_train_fold, X_val_fold = X_train_pca[train_idx], X_train_pca[val_idx]
    y_train_fold, y_val_fold = y_train_full[train_idx], y_train_full[val_idx]

    model_fold = clone(best_model)
    model_fold.fit(X_train_fold, y_train_fold)

    val_acc = model_fold.score(X_val_fold, y_val_fold)
    logger.info(f"Fold validation accuracy: {val_acc:.5f}")

    if val_acc > best_score:
        best_score = val_acc
        best_model = model_fold  # Store best-trained model

logger.info(f"Best fold validation accuracy: {best_score:.5f}")

# Train the Final Model on the Full Training Data
logger.info("Training Final Random Forest Model...")
best_model.fit(X_train_pca, y_train_full)

# Final Testing
logger.info("Evaluating on Test Set...")
y_pred = best_model.predict(X_test_pca)

# Convert Index Labels to Emotion Names
y_pred_labels = [emotion_labels[i] for i in y_pred]
y_test_labels = [emotion_labels[i] for i in y_test]

# Generate Metrics
classification = classification_report(y_test_labels, y_pred_labels, target_names=emotion_labels)
conf_matrix = confusion_matrix(y_test_labels, y_pred_labels)

# Log Results
logger.info("Classification Report:\n" + classification)
logger.info("Confusion Matrix:\n" + str(conf_matrix))

# Compute Accuracy
train_acc = best_model.score(X_train_pca, y_train_full)
test_acc = best_model.score(X_test_pca, y_test)

logger.info(f"Training Accuracy: {train_acc:.5f}")
logger.info(f"Test Accuracy: {test_acc:.5f}")
