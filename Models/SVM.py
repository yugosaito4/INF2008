import os
import logging
import time
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import psutil  # For logging memory usage

# **Set Up Logging**
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/svm_training_ckplus.log",
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
def log_and_print(message):
    """Helper function to log and print messages."""
    print(message)
    logging.info(message)

def log_memory_usage():
    """Logs the current memory usage."""
    process = psutil.Process(os.getpid())
    mem_usage = process.memory_info().rss / (1024 ** 2)  # Convert to MB
    log_and_print(f"Memory Usage: {mem_usage:.2f} MB")

log_and_print("Starting SVM Training for CK+...")

# **Load Extracted Features**
input_parquet = "CSV/ckplus_features.parquet"  # Use CK+ features
if not os.path.exists(input_parquet):
    raise FileNotFoundError(f"Feature file {input_parquet} not found!")

df = pd.read_parquet(input_parquet)
log_and_print(f"Loaded dataset with {df.shape[0]} samples and {df.shape[1]} features.")

# **Separate Features and Labels**
X = df.drop(columns=["label"]).values
y = df["label"].values

# **Encode Labels**
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# **Split Dataset (80% Train, 20% Test)**
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded  # No SMOTE needed
)
log_and_print(f"Split dataset: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples.")
log_memory_usage()

# **GridSearchCV for SVM Optimization (No Class Weights)**
log_and_print("Starting GridSearchCV for hyperparameter tuning on CK+...")
start_time = time.time()

param_grid = {
    "C": [10],  # **Expanded C range**
    "gamma": ["scale", "auto" , 0.01 , 0.001],  # Test both scale and auto gamma
    "kernel": ["rbf"]
}

grid_search = GridSearchCV(
    SVC(), param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=2  
)
grid_search.fit(X_train, y_train)

best_svm = grid_search.best_estimator_
best_params = grid_search.best_params_
log_and_print(f"Best SVM Parameters for CK+: {best_params}")

end_time = time.time()
log_and_print(f"GridSearchCV completed in {end_time - start_time:.2f} seconds.")
log_memory_usage()


# **Evaluate Best Model on Train Set**
train_accuracy = best_svm.score(X_train, y_train)  # Training accuracy
log_and_print(f"Training Accuracy: {train_accuracy:.4f}")

# **Evaluate Best Model on Test Set**
log_and_print("Evaluating the best SVM model on the test set...")
y_pred = best_svm.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)  # Test accuracy
log_and_print(f"Test Accuracy: {test_accuracy:.4f}")

# **Print Classification Report**
class_report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
log_and_print("\nClassification Report:\n" + class_report)

# **Confusion Matrix**
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.savefig("logs/confusion_matrix_ckplus.png")
plt.show()

log_and_print("Confusion matrix saved to logs/confusion_matrix_ckplus.png.")
log_memory_usage()