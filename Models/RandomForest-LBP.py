# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from scipy.stats import randint

# Load dataset
data = pd.read_csv("LBP_Extraction/lbp_features_all_emotions.csv")

# Ensure all features are numeric (in case of hidden non-numeric values)
for col in data.columns[:-1]:  # Excluding label
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Encode labels correctly
label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['label'])

# Split dataset
X = data.iloc[:, :-1].values  # Features
y = data['label'].values     

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, random_state=42, stratify=y)

# Define RandomForest hyperparameters
param_dist = {
    'n_estimators': randint(50, 200),
    'max_depth': randint(5, 20),
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 5)
}

# Initialize RandomForestClassifier
rf = RandomForestClassifier(random_state=42)

# Hyperparameter tuning with RandomizedSearchCV
random_search = RandomizedSearchCV(
    rf,
    param_distributions=param_dist,
    n_iter=10,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)

# Fit the model
random_search.fit(X_train, y_train)

# Predict on test data
y_pred = random_search.best_estimator_.predict(X_test)

# Evaluate model performance
classification_rep = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Classification Report:\n", classification_rep)
print("Confusion Matrix:\n", conf_matrix)
