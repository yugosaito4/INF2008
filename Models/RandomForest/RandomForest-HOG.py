import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Load dataset
logger.info("Loading dataset...")
data = pd.read_csv("HOG_Extraction/hog_facial_expression_features_250.csv")

X = data.iloc[:, 1:].values  # All columns except the first ('label')
y = data['label'].values      # The 'label' column
logger.info(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features.")

# Define Stratified K-Fold Cross-Validation (Preserves class distribution)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Hyperparameters Adjusted for tuning
model = RandomForestClassifier(
    n_estimators=500,       # Reduce number of trees
    max_depth=15,           # Reduce tree depth to prevent memorization
    min_samples_split=20,   # Require more samples before a split
    min_samples_leaf=10,    # Require more samples in leaf nodes
    max_features='log2',    # Consider fewer features per split
    class_weight='balanced',  # Keep to handle class imbalance
    bootstrap=True,         # Keep bootstrapping for generalization
    random_state=42,
    n_jobs=-1
)

# Perform Stratified K-Fold Cross-Validation
logger.info("Performing 5-Fold Stratified Cross-Validation...")
cv_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy', n_jobs=-1)

# Log Cross-validation results
logger.info(f"Stratified Cross-validation accuracy: {cv_scores}")
logger.info(f"Mean CV Accuracy: {cv_scores.mean():.5f} ± {cv_scores.std():.5f}")

# Final Training on Full Dataset and Test Set Evaluation
logger.info("Training the Random Forest model on full training data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42, stratify=y)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Classification report
classification = classification_report(y_test, y_pred)
logger.info("Classification Report:\n" + classification)

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
logger.info("Confusion Matrix:\n" + str(conf_matrix))

# Compute training accuracy
train_acc = model.score(X_train, y_train)

# Compute test accuracy
test_acc = model.score(X_test, y_test)

logger.info(f"Training Accuracy: {train_acc:.5f}")
logger.info(f"Test Accuracy: {test_acc:.5f}")
