import lightgbm as lgb
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
optimized_data = pd.read_parquet("C:/Users/denni/Downloads/optimized_features.parquet")

# ✅ Extract features and labels (KEEP AS DATAFRAME)
X_combined = optimized_data.iloc[:, 1:]  # ✅ REMOVED `.values` to keep column names
y, emotion_labels = optimized_data["label"].factorize()  # Encode emotion labels

logger.info(f"Emotion Labels Mapping: {dict(enumerate(emotion_labels))}")

# ✅ Split dataset (80% train, 20% test) (STILL A DATAFRAME)
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_combined, y, test_size=0.2, random_state=42, stratify=y
)

# ✅ Train LightGBM with Gradient Boosting
best_model = lgb.LGBMClassifier(
    boosting_type="gbdt",  # ✅ SWITCHED TO GRADIENT BOOSTING
    n_estimators=2000,  # ✅ More boosting rounds for better learning
    learning_rate=0.01,  # ✅ Controls step size (lower = better generalization)
    max_depth=12,  # ✅ Restrict depth to prevent overfitting
    num_leaves=64,  # ✅ More leaves for complex splits
    feature_fraction=0.8,  # ✅ Randomly select 80% features for each tree
    bagging_fraction=0.8,  # ✅ Subsample 80% data for each boosting iteration
    bagging_freq=5,  # ✅ Perform bagging every 5 iterations
    min_data_in_leaf=10,  # ✅ Avoid creating leaves with very few samples
    class_weight="balanced",  # ✅ Handle class imbalance
    reg_alpha=0.1,  # ✅ L1 Regularization (prevents overfitting)
    reg_lambda=0.1,  # ✅ L2 Regularization (prevents overfitting)
    random_state=42,
    #n_jobs=-1,
    
    # ✅ GPU SETTINGS (Enable if GPU is available)
    device="cpu",  # Change to "gpu" if needed
    gpu_use_dp=False,  # Use single precision for speed (if using GPU)
    
    # ✅ Early stopping
    early_stopping_rounds=100  # ✅ Stops if performance doesn't improve
)

logger.info("Training the final LightGBM Gradient Boosting model...")
best_model.fit(
    X_train_full, y_train_full, 
    eval_set=[(X_test, y_test)], 
    eval_metric="logloss",
    callbacks=[lgb.log_evaluation(period=100)]  # Logs every 100 iterations
)

# ✅ Evaluate on test set
logger.info("Evaluating on test set...")
y_pred = best_model.predict(X_test)  # ✅ No more warnings!

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
