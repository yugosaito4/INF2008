import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix
import logging
import collections

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

### **1️⃣ Load and Normalize HOG + LBP Features from Separate Files**
logger.info("Loading HOG and LBP features from separate files...")

hog_data = pd.read_csv("HOG_Extraction/hog_facial_expression_features_120.csv")  # HOG features
lbp_data = pd.read_csv("LBP_Extraction/lbp_features_all_emotions.csv")  # LBP features

assert hog_data.shape[0] == lbp_data.shape[0], "Mismatch in number of samples between HOG and LBP"

hog_features = hog_data.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').values
lbp_features = lbp_data.iloc[:, :-1].apply(pd.to_numeric, errors='coerce').values

# Normalize using MinMaxScaler instead of StandardScaler
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

### **2️⃣ Perform Stratified K-Fold Cross-Validation**
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define optimized Random Forest model
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

logger.info("Performing 5-Fold Stratified Cross-Validation...")
cv_scores = cross_val_score(model, X_combined, y, cv=skf, scoring='accuracy', n_jobs=-1)

logger.info(f"Mean CV Accuracy: {cv_scores.mean():.5f} ± {cv_scores.std():.5f}")

### **3️⃣ Final Training and Testing**
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.3, shuffle=True, random_state=42, stratify=y)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_pred_labels = label_encoder.inverse_transform(y_pred)
y_test_labels = label_encoder.inverse_transform(y_test)

# Classification Report & Confusion Matrix
classification = classification_report(y_test_labels, y_pred_labels)
logger.info("Classification Report:\n" + classification)

conf_matrix = confusion_matrix(y_test_labels, y_pred_labels)
logger.info("Confusion Matrix:\n" + str(conf_matrix))

# Accuracy
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)
logger.info(f"Training Accuracy: {train_acc:.5f}")
logger.info(f"Test Accuracy: {test_acc:.5f}")
