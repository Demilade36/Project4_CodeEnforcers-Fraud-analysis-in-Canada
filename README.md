
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

## 📊 Project Overview  

We trained classification models using cleaned and labeled fraud records.:

- 🔹 **Fraud Classification**  
  Predict whether a report represents fraud (1) or non-fraud (0), based on categorical and numeric indicators.

- 🔹 **Additional**  
  Focus on confirmed fraud reports and predict whether a victim lost money (1) or did not (0).

All data operations are supported by **SQLite3**, visualized with **Matplotlib**, and modeled using **scikit-learn** tools.

---

## 🌐 Data Source Access/Overview

Retrieved fraud report data directly from the [Canada’s Open Government Portal-Canadian Anti-Fraud Centre Fraud Reporting System](https://open.canada.ca/data/en/dataset/6a09c998-cddb-4a22-beff-4dca67ab892f) using their public API.


### 🗃️ Dataset Overview  

**📁 Source:** Canadian Anti-Fraud Centre Reporting Data (2021–2025)  
The dataset includes case-level fraud details with demographic, financial, and geographic attributes.

| Feature              | Description                                                       |
|----------------------|-------------------------------------------------------------------|
| `Number ID`          | Unique case ID                                                    |
| `Date Received`      | Complaint filing date                                             |
| `Province`           | Canadian province where reported                                  |
| `Complaint Type`     | 1 = Fraud, 0 = Attempt/Other/Unknown                              |
| `Gender`             | Male, Female, Prefer not to say, Not Available                    |
| `Victim Age Range`   | Age group (e.g., 20–29, 30-39, 60-69)                             |
| `Fraud Category`     | Scam type: investment, merchandise, romance, etc.                |
| `Solicitation Method`| Contact method: phone, email, social media, etc.                 |
| `Number of Victims`  | 1 = Victim, 0 = Non-victim                                        |
| `Dollar Loss`        | Reported financial loss (if any)                                  |


## ⚙️ Methodology & Tools  

| Stage               | Description                                                      |
|---------------------|------------------------------------------------------------------|
| Cleaning & Filtering| Removed missing values, filtered for Canada-only                |
| Encoding            | Label encoding for model-ready categorical features             |
| Database            | SQLite3 used for querying and storage                           |
| Modeling            | Random Forest & K-Nearest Neighbors (KNN) classifiers           |
| Visualization       | Used Matplotlib for bar charts, pie charts and trend visuals   |
| Evaluation          | Metrics include Accuracy, Precision, Recall, and F1 Score       |

---

## 🎯 Key Findings  

- **Top Provinces for Fraud:** Ontario, Quebec, British Columbia  
- **Common Fraud Types:** Merchandise fraud, Investment scams, Identity theft  
- **High-Risk Age Groups:** 30–39 and 60–69 were among the most targeted  

### 💰 Financial Impact Trends:

| Year | Reported Fraud Loss |
|------|----------------------|
| 2021 | $383 million         |
| 2022 | $530 million         |
| 2023 | $569 million         |
| 2024 | $638 million         |

---

## 📊 Visualizations

---
<!-- ![Contract Type Churn Rate](https://github.com/Eder-2024/Project_1-Telco/blob/main/Plot/Contract%20Type%20Churn%20Rate.png)
<!-- **H1: Fraud Victimization Peaks Among Adults Aged 30–39 (Accepted)**
![Contract Type Churn Rate](./Plot/Contract%20Type%20Churn%20Rate.png)

Confusion Matrix
Decision Boundary of Fraud Detection Model
![Fraud vs ]
![Victims by Age Group – Bar Chart](https://github.com/your-username/your-repo/blob/main/visuals/Victims_by_Age_Group.png)

![Fraud Loss by Age Group – Box Plot](https
Decision Boundary of Fraud Detection Model://github.com/your-username/your-repo/blob/main/visuals/Loss_by_Age_Group.png) -->

![Victim Count by Year and Age – Trend Line](https://github.com/your-username/your-repo/blob/main/visuals/Victims_Trend_by_Age.png)

---


## 📈 Model Performance 

| Model                         | Dataset                    | Accuracy | Precision | Recall | F1 Score |
|-------------------------------|----------------------------|----------|-----------|--------|----------|
| Random Forest Classifier      | Fraud vs. No Fraud         | 89%      | 87%       | 86%    | 86.5%    |
| K Neighbors Classifier (K=15) | Fraud vs. No Fraud         | 83%      | 81%       | 79%    | 80%      |



---

## 🧑‍💼 Stakeholder Impact

| Stakeholder         | Impact                                                                 |
|---------------------|------------------------------------------------------------------------|
| Law Enforcement     | Identify fraud hotspots and enhance preventive actions                  |
| Financial Institutions | Enhance fraud risk scoring and alert systems                      |
| Public & Media      | Inform campaigns based on fraud type, method, and geography            |
| Policy Makers       | Use data to allocate resources and design education initiatives        |

---

## 📁 Project Structure

```
Project4_CodeEnforcers-Fraud-analysis-in-Canada/
│
├── Plot/                       # Visualizations and charts
├── fraud_data.ipynb            # Data cleaning and preprocessing
├── Machine_learning_new.ipynb  # Model building and evaluation
├── model_evaluation.xlsx       # Excel file with performance metrics
├── Project 4 - Project Proposal.docx  # Project planning document
└── README.md                   # Executive summary and documentation
```

---`

## 🛠️ Libraries Used

* **scikit-learn** – Machine learning (data prep, modeling, evaluation)
* **matplotlib** – Plotting graphs
* **pandas, numpy** – Data handling and math
* **sqlite3, json** – Database and data parsing
* **urllib, time** – Data fetching and timing

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

## 🙌 Team – Group 4: *The Code Enforcers*

* **Eder Ortiz** – data cleaning, database setup, model development, metric analysis
* **Oludemilade Adenuga** – model development, metric analysis, and documentation
* **Geraldine Valencia** – visualization, data evaluation, and documentation

---

## 📚 References

* **Data Source:** [Canadian Anti-Fraud Centre – Open Canada](https://open.canada.ca)
* **scikit-learn Documentation:** [https://scikit-learn.org](https://scikit-learn.org)
* **Matplotlib Documentation:** [https://matplotlib.org](https://matplotlib.org)
* **SQLite3 Documentation:** [Python SQLite3 Module](https://docs.python.org/3/library/sqlite3.html)

---

> “We use models to predict fraud, but the mission is always about protecting people.”
> – *The Code Enforcers*

```


