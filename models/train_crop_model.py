from sklearn.ensemble import RandomForestClassifier
#from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "Crop_recommendation.csv")
data = pd.read_csv(DATA_PATH)
print("✅ Loaded dataset from:", DATA_PATH)


X = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = data['label']  # should be crop names
print("Data shape:", data.shape)
print(data.head())
print(data['label'].value_counts())


# Encode crop labels to numbers
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

#xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
#xgb_model.fit(X_train, y_train)

# Save models
joblib.dump(rf_model, 'rf_model.joblib')
#joblib.dump(xgb_model, 'xgb_model.joblib')
joblib.dump(encoder, 'label_encoder.joblib')
print("RF Accuracy:", rf_model.score(X_test, y_test))
# print("XGB Accuracy:", xgb_model.score(X_test, y_test))
