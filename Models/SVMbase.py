import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

logger.info("Loading dataset...")
data = pd.read_csv("LBP_Extraction/lbp_features_all_emotions.csv")


X = data.iloc[:, :-1].values  # All columns except the last ('label')
y = data['label'].values      # The 'label' column
logger.info(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features.")

#splitting into training and testing , 8:2
logger.info("Splitting dataset into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)
logger.info(f"Training set: {X_train.shape[0]} samples. Testing set: {X_test.shape[0]} samples.")

logger.info("Training the base SVM model")
model = SVC(class_weight='balanced') #balance the dataset



model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Classification report
classification = classification_report(y_test, y_pred)
logger.info("Classification Report:\n" + classification)

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
logger.info("Confusion Matrix:\n" + str(conf_matrix))
