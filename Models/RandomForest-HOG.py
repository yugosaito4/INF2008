import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from scipy.stats import randint
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Load dataset
logger.info("Loading dataset...")
data = pd.read_csv("HOG_Extraction/hog_facial_expression_features.csv")

X = data.iloc[:, 1:].values  # All columns except the first ('label')
y = data['label'].values      # The 'label' column
logger.info(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features.")

# Split dataset (80% training, 20% testing)
logger.info("Splitting dataset into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42, stratify=y)
logger.info(f"Training set: {X_train.shape[0]} samples. Testing set: {X_test.shape[0]} samples.")

# Define hyperparameter search space
param_dist = {
    'n_estimators': randint(50, 200),
    'max_depth': [10, 20, None],
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 4),
    'max_features': ['sqrt', 'log2', None]
}

# Perform Randomized Search
logger.info("Tuning hyperparameters using Randomized Search...")
random_search = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_dist, n_iter=20, cv=3, verbose=0, n_jobs=-1, random_state=42)
random_search.fit(X_train, y_train)

# Get best parameters
best_params = random_search.best_params_
logger.info(f"Best Parameters: {best_params}")

# Train the final Random Forest model with best parameters
logger.info("Training the optimized Random Forest model...")
model = RandomForestClassifier(**best_params, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Classification report
classification = classification_report(y_test, y_pred)
logger.info("Classification Report:\n" + classification)

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
logger.info("Confusion Matrix:\n" + str(conf_matrix))
