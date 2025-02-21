import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
import logging
import os, psutil


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Load HOG feature dataset
logger.info("Loading dataset...")
data = pd.read_csv("HOG_Extraction/hog_facial_expression_features_120.csv")


X = data.iloc[:, 1:].values  # all columns except the first (features)
y = data.iloc[:, 0].values  # label column (emotion)

logger.info(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features.")

#splitting into training and testing , 8:2
logger.info("Splitting dataset into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)
logger.info(f"Training set: {X_train.shape[0]} samples. Testing set: {X_test.shape[0]} samples.")

# feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# reduce features using PCA
pca = PCA(n_components=85)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
logger.info(f"PCA applied: New feature size = {X_train_pca.shape[1]}")

# use SMOTE to balance the classes in training data
os.environ["LOKY_MAX_CPU_COUNT"] = str(psutil.cpu_count(logical=False))
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_pca, y_train)
logger.info(f"After SMOTE: Training set now has {X_train_bal.shape[0]} samples.")

# Train the SVM model
logger.info("Training the SVM model")
model = SVC(C=1.2, gamma=0.008, kernel='rbf', class_weight='balanced')
model.fit(X_train_bal, y_train_bal)

# Predict on training set
y_train_pred = model.predict(X_train_bal)
train_accuracy = accuracy_score(y_train_bal, y_train_pred)
logger.info(f"Training Accuracy: {train_accuracy:.4f}")

# Predict on test set
y_pred = model.predict(X_test_pca)
test_accuracy = accuracy_score(y_test, y_pred)
logger.info(f"Test Accuracy: {test_accuracy:.4f}")

# Classification report
classification = classification_report(y_test, y_pred)
logger.info("Classification Report:\n" + classification)

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
logger.info("Confusion Matrix:\n" + str(conf_matrix))
