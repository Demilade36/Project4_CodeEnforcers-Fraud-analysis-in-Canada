
# 💸 Fraud Analysis in Canada (2021–2025)  
**Group 4 – The Code Enforcers 🕵️‍♂️📊**

<p align="center">
  <img src="https://img.shields.io/badge/Pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/NumPy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/SQLite3-%234B6E60.svg?style=for-the-badge&logo=sqlite&logoColor=white" />
  <img src="https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black" />
  <img src="https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Python%203.9+-blue.svg?style=for-the-badge&logo=python&logoColor=white" />
</p>

---

## 💡 Introduction  
Since 2021, Canadians have reported over **$2 billion** in fraud-related losses to the Canadian Anti-Fraud Centre (CAFC). With increasing complexity in scam tactics and a steady rise in victims, fraud has become a national crisis.  

This project leverages **machine learning** to classify fraud vs. non-fraud cases and analyze **dollar loss impact**, using CAFC data from 2021 to 2025.  

> “We can’t prevent every scam—but we can use data to detect and disrupt fraud faster.” – *The Code Enforcers*

---

## 📊 Project Objective

- Train the model to know when there is Fraud or No Fraud. No Fraud includes an attempt, other or unknown. As code enforcers, we assumed these three categories as 'No Fraud'. We included Optimization of the KNN model.

- **Additional:** Decided to build a predictive interactive model for users. This model is running on victims only that reported fraud in Canada and will tell the user a situation where there is a possiblity of financial loss.
---

## 🗃️ Dataset  Overview
**Source :** Retrieved fraud report data directly from the [Canada’s Open Government Portal-Canadian Anti-Fraud Centre Fraud Reporting Data (2021-2025)](https://open.canada.ca/data/en/dataset/6a09c998-cddb-4a22-beff-4dca67ab892f) using their public API.

The dataset includes case-level fraud details with demographic, financial, and geographic attributes.

---

## ⚙️ Methodology & Tools:

| **Stage**            | **Description**                                                                 | **Tools / Libraries Used** |
| -------------------- | ------------------------------------------------------------------------------- | -------------------------- |
| Data Fetching        | Retrieved data via API and managed request timing                               | `urllib`, `time`           |
| Cleaning & Filtering | Removed missing values; filtered data for Canada-only                           | `pandas`, `numpy`          |
| Encoding             | Applied label encoding to categorical features for model compatibility          | `scikit-learn`             |
| Database             | Used SQLite3 for data querying and storage                                      | `sqlite3`, `json`          |
| Visualization        | Created bar charts, pie charts, and trend visuals to illustrate key patterns    | `matplotlib`               |
| Modeling             | Random Forest and K-Nearest Neighbors (KNN) classifiers                         | `scikit-learn`             |
| Evaluation           | Measured performance using Accuracy, Precision, Recall, and F1 Score            | `scikit-learn`             |

---

## 🎯 Key Findings  
### Fraud vs No Fraud Cases
  <img src="./Plot/Fraud%20vs%20No%20Fraud%20Cases.png" alt="Fraud vs No Fraud Cases" width="400"/>

### Victims of Fraud by Province
  <img src="./Plot/Victims%20by%20Province.png" alt="Victims by Province" width="400"/>

### Victims of Fraud by Age range
  <img src="./Plot/Victims%20by%20Age%20Range.png" alt="Victims by Age Range" width="400"/>


### 💰 Financial Impact Trends:
| Year | Reported Fraud Loss |
|------|----------------------|
| 2021 | $388 million         |
| 2022 | $533 million         |
| 2023 | $577 million         |
| 2024 | $647 million         |

**Top Provinces for Fraud:**   Ontario, Quebec, British Columbia  
**Fraud Types with Top Dollar Loss :**   Investment scams, Romance scams, and Spear Phishing   
**High-Risk Age Groups:**   20-29, 30–39 and 60–69 were among the most targeted  

---

## ✅ Model Performance: 

| Metric       | Random Forest        |               | KNN (K = 15)         |               |
|--------------|----------------------|---------------|----------------------|---------------|
|              | No Fraud (0)         | Fraud (1)     | No Fraud (0)         | Fraud (1)     |
| **Precision**| 0.78                 | 0.94          | 0.76                 | 0.92          |
| **Recall**   | 0.87                 | 0.88          | 0.84                 | 0.88          |
| **F1 Score** | 0.82                 | 0.91          | 0.80                 | 0.90          |
| **Accuracy** | 0.88                 |               | 0.87

### 🧮 Confusion Matrix :

| Prediction \ Actual | No Fraud (0) | Fraud (1)   |   | Prediction \ Actual | No Fraud (0) | Fraud (1)   |
| ------------------- | ------------ | ----------- | - | ------------------- | ------------ | ----------- |
| **Random Forest**   |              |             |   | **KNN (K=15)**      |              |             |
| **No Fraud (0)**    | 13,762 (TN)  |  2,018 (FP) |   | **No Fraud (0)**    | 13,280 (TN)  |  2,500 (FP) |
| **Fraud (1)**       |  3,959 (FN)  | 30,077 (TP) |   | **Fraud (1)**       |  4,090 (FN)  | 29,946 (TP) |

### Key:
- **TN (True Negative):** Correctly predicted as "No Fraud."
- **FP (False Positive):** Incorrectly flagged as fraud.
- **FN (False Negative):** Fraud cases missed.
- **TP (True Positive):** Correctly identified as fraud.

### Insights: 
- **Random Forest** outperforms KNN in **precision**, especially in detecting fraud (0.94 vs. 0.92).
- **Recall** values are similar for both models, but Random Forest slightly edges out in identifying non-fraud cases.
- **F1 Score** and **overall accuracy** show Random Forest performing slightly better (**0.88 vs. 0.87**).
- **Confusion matrix** suggests Random Forest makes fewer false fraud classifications.
- Random Forest appears to be a more **balanced model** for fraud detection, achieving **higher precision** while maintaining competitive recall and accuracy compared to KNN.

---

## 📁 Project Structure

Project4_CodeEnforcers-Fraud-analysis-in-Canada/
│
├── Plot/                       # Visualizations and charts
├── fraud_data.db               # Database
├── Machine_learning.ipynb      # Data cleaning, model building, and evaluation
├── model_evaluation.xlsx       # Actual vs predicted class labels
├── Project 4 - Project Proposal.docx  # Project planning document
└── README.md                   # Executive summary and documentation

---

## 🛠️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Demilade36/Project4_CodeEnforcers-Fraud-analysis-in-Canada.git
cd Project4_CodeEnforcers-Fraud-analysis-in-Canada
```

---
### 2. Install Required Python Libraries

Make sure you have **Python 3.9+** and **pip** installed. Then run:

```bash
pip install pandas numpy matplotlib scikit-learn
```
---

## 🚧 Dataset Limitations

* **Underreporting:** Many victims never report, especially small losses
* **Time Lag:** Delays between incident and report date
* **Simplification:** Non-fraud includes multiple non-harmful categories
* **Geographic Bias:** Ontario dominates reports due to population density
---

## 🧭 Ethical Considerations

* No personal identifiers included
* Data used solely for public interest, research, and education
* Respect for data privacy and contextual sensitivity around victimization
---

## 🧑‍💼 Stakeholder Impact

| Stakeholder         | Impact                                                                 |
|---------------------|------------------------------------------------------------------------|
| Law Enforcement     | Identify fraud hotspots and enhance preventive actions                  |
| Financial Institutions | Enhance fraud risk scoring and alert systems                      |
| Public & Media      | Inform campaigns based on fraud type, method, and geography            |
| Policy Makers       | Use data to allocate resources and design education initiatives        |
---

## 📚 References

* **scikit-learn Documentation:** [https://scikit-learn.org](https://scikit-learn.org)
* **Matplotlib Documentation:** [https://matplotlib.org](https://matplotlib.org)
* **pandas Documentation:** [https://pandas.pydata.org](https://pandas.pydata.org)
* **NumPy Documentation:** [https://numpy.org](https://numpy.org)
* **SQLite3 Documentation:** [https://docs.python.org/3/library/sqlite3.html](https://docs.python.org/3/library/sqlite3.html)
* **JSON Module:** [https://docs.python.org/3/library/json.html](https://docs.python.org/3/library/json.html)
* **urllib Module:** [https://docs.python.org/3/library/urllib.html](https://docs.python.org/3/library/urllib.html)
* **time Module:** [https://docs.python.org/3/library/time.html](https://docs.python.org/3/library/time.html)
---

## 🙌 Team – Group 4: *The Code Enforcers*

* **Eder Ortiz** – data cleaning, database setup, model development and  metric analysis
* **Oludemilade Adenuga** – model development, metric analysis, and documentation
* **Geraldine Valencia** – visualization, data evaluation, and documentation
---
## 🏛️ License
This project is licensed under the **Open Government Licence - Canada**.  
For more details, see: [Open Government Licence - Canada](https://open.canada.ca/en/open-government-licence-canada).
