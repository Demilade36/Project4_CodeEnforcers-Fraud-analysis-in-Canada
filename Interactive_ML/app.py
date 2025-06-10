from flask import Flask, request, render_template, jsonify
import sqlite3
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

app = Flask(__name__)

# Load Data
conn = sqlite3.connect("../fraud_data.db")
query = "SELECT * FROM fraud_reports"
df = pd.read_sql(query, conn)
conn.close()
df_canada = df[df["Country"] == "Canada"].copy()
df_canada= df_canada[df_canada["Number of Victims"] > 0]
df_canada = df_canada.dropna()
df=df_canada.copy()

# Feature Encoding
categorical_columns = ["Complaint Received Type", "Province/State", "Fraud and Cybercrime Thematic Categories", "Solicitation Method", "Gender", "Language of Correspondence","Victim Age Range"]
label_encoders = {}
for col in categorical_columns:
    label_encoders[col] = LabelEncoder()
    df[col] = label_encoders[col].fit_transform(df[col])

# Feature Selection & Scaling
X = df.drop(["Number ID", "Dollar Loss","Date Received","Country","Complaint Type","Number of Victims"], axis=1)
X_numeric = X.select_dtypes(include=['number'])  # Select only numeric columns
y = df["Dollar Loss"].apply(lambda x: 1 if x > 0 else 0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train ML Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)



@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        # Extract user input
        user_data = {}
        for col in X.columns:
            value = request.form[col]
            if col in label_encoders:
                # Encode categorical value using the pre-trained encoder
                value = label_encoders[col].transform([value])[0]
            else:
                value = float(value)  # Handle numeric inputs if any
            user_data[col] = [value]

        user_df = pd.DataFrame(user_data)
        user_scaled = scaler.transform(user_df)
        prediction = model.predict(user_scaled)[0]
        return render_template("index.html", prediction=prediction)

    return render_template("index.html", prediction=None)

@app.route("/graphs", methods=["GET"])
def show_graphs():
    graphs = ["Fraud Distribution", "Feature Importance", "PCA Visualization", "Confusion Matrix"]
    return render_template("graphs.html", graphs=graphs)

if __name__ == "__main__":
    app.run(debug=True)