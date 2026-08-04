import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

print("Current Working Directory:", os.getcwd())

# Load the dataset
data = pd.read_csv("../data/Fertilizer Prediction.csv")

# 🔧 Clean column names (remove all spaces)
data.columns = data.columns.str.strip()

# Check columns again
print("Cleaned Columns:", data.columns.tolist())

# ✅ Use correct target column (after stripping spaces)
target_col = "Fertilizer Name"
if target_col not in data.columns:
    # find similar column names
    for col in data.columns:
        if "Fertilizer" in col:
            target_col = col
            print(f"Using detected column name: {target_col}")
            break

# Split features and target
X = data.drop(target_col, axis=1)
y = data[target_col]

# Convert categorical features to numeric
X = pd.get_dummies(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
print("✅ Fertilizer Model Accuracy:", accuracy)

# Save the model

pickle.dump(model, open("fertilizer_model.pkl", "wb"))
print("💾 Model saved successfully as 'models/fertilizer_model.pkl'")
