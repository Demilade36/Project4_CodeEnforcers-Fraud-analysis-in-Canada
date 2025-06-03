# Interactive ML on CAFC dataset (Supervised + Unsupervised + Regression + Visualization)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Load and clean data
@st.cache_data

def load_data():
    df = pd.read_csv("/mnt/data/cafc-open-gouv-database-2021-01-01-to-2025-03-31-extracted-2025-04-01.csv")

    # Drop French duplicates and ID
    df = df.drop(columns=[
        'Type de plainte reçue', 'Pays', 'Province/État',
        'Catégories thématiques sur la fraude et la cybercriminalité',
        'Méthode de sollicitation', 'Genre', 'Langue de correspondance', 'Type de plainte',
        'Numéro d'identification / Number ID'
    ])

    # Clean dollar loss
    df['Dollar Loss /pertes financières'] = (
        df['Dollar Loss /pertes financières']
        .replace('[\$,]', '', regex=True)
        .astype(float)
    )

    # Fill NA with 'Unknown'
    df = df.fillna('Unknown')

    return df

df = load_data()

# Encode categorical features
cat_cols = ['Complaint Received Type', 'Country', 'Province/State',
            'Fraud and Cybercrime Thematic Categories', 'Solicitation Method', 'Gender',
            'Language of Correspondence', "Victim Age Range / Tranche d'âge des victimes"]

df_encoded = df.copy()
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Supervised learning - Classification
X = df_encoded[cat_cols]
y_class = df_encoded['Complaint Type']

X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X, y_class, stratify=y_class, test_size=0.2, random_state=42)
scaler_class = StandardScaler()
X_train_class_scaled = scaler_class.fit_transform(X_train_class)
X_test_class_scaled = scaler_class.transform(X_test_class)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_class_scaled, y_train_class)
supervised_accuracy = clf.score(X_test_class_scaled, y_test_class)

# Supervised learning - Regression (Dollar Loss)
y_reg = df_encoded['Dollar Loss /pertes financières']
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)
scaler_reg = StandardScaler()
X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
X_test_reg_scaled = scaler_reg.transform(X_test_reg)

regressor = LinearRegression()
regressor.fit(X_train_reg_scaled, y_train_reg)
regression_score = regressor.score(X_test_reg_scaled, y_test_reg)

# Unsupervised learning - KMeans
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans_labels = kmeans.fit_predict(X_pca)

# Streamlit UI
st.title("📊 CAFC Interactive Fraud Prediction App")
st.write("Use the form below to simulate and predict outcomes based on user input.")

with st.form("user_input_form"):
    input_data = {}
    for col in cat_cols:
        options = df[col].unique().tolist()
        selected = st.selectbox(col, options)
        input_data[col] = selected

    submitted = st.form_submit_button("Predict")

if submitted:
    # Prepare input
    input_vector = np.array([
        label_encoders[col].transform([input_data[col]])[0] for col in cat_cols
    ]).reshape(1, -1)

    # Predictions
    class_pred = clf.predict(scaler_class.transform(input_vector))[0]
    loss_pred = regressor.predict(scaler_reg.transform(input_vector))[0]
    cluster = kmeans.predict(pca.transform(input_vector))[0]

    st.subheader("🔍 Prediction Result")
    st.write(f"**Complaint Type Prediction:** {class_pred}")
    st.write(f"**Estimated Dollar Loss:** ${loss_pred:,.2f}")
    st.write(f"**Cluster Assignment:** Cluster {cluster}")
    st.write(f"**Classification Accuracy:** {supervised_accuracy * 100:.2f}%")
    st.write(f"**Regression R² Score:** {regression_score:.2f}")

    st.info("Note: This is a simulated prediction based on historical data from CAFC.")

# Cluster Visualization
st.subheader("📌 Fraud Case Cluster Visualization")
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=kmeans_labels, palette="Set2", s=30)
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("KMeans Clusters of Fraud Cases")
st.pyplot(fig)
