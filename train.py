import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import pickle
import os

# Load dataset
df = pd.read_csv("data/loan_data.csv")

# 🔥 Fix: remove all spaces from column names
df.columns = df.columns.str.strip()

print("CLEAN COLUMNS:", df.columns.tolist())

# Handle missing values
df.ffill(inplace=True)

# Convert categorical to numeric
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

# Features and target
X = df.drop("loan_status", axis=1)
y = df["loan_status"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Create model folder
os.makedirs("model", exist_ok=True)

# Save model
pickle.dump(model, open("model/model.pkl", "wb"))

print("\n✅ Model trained and saved successfully!")