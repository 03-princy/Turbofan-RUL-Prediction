# ✈️ Turbofan Engine Remaining Useful Life (RUL) Prediction

## 🚀 Project Overview

Predictive Maintenance is one of the most important applications of Artificial Intelligence in the aerospace industry.

This project predicts the **Remaining Useful Life (RUL)** of turbofan aircraft engines using the **NASA CMAPSS FD001 dataset**. The system analyzes engine operational settings and sensor measurements to estimate how many cycles remain before an engine requires maintenance.

The project includes:

* Data Preprocessing & Feature Engineering
* Machine Learning Model Training
* Model Comparison (Random Forest vs XGBoost)
* Performance Evaluation
* Interactive Streamlit Dashboard
* Real-time RUL Prediction
* Engine Health Monitoring

---

## 🎯 Problem Statement

Aircraft engines generate large amounts of sensor data during operation.

The goal is to predict:

> **How many cycles remain before an engine is likely to fail?**

Accurate prediction helps:

* Reduce maintenance costs
* Prevent unexpected failures
* Improve operational safety
* Enable predictive maintenance

---

## 📊 Dataset

### NASA CMAPSS Turbofan Engine Dataset

Source: NASA Prognostics Center of Excellence

Files Used:

```text
train_FD001.txt
test_FD001.txt
RUL_FD001.txt
```

Dataset contains:

* Engine ID
* Operational Settings
* 21 Sensor Measurements
* Engine Cycles

Total Features Used:

```text
3 Operational Settings
21 Sensor Readings
-------------------
24 Input Features
```

---

## ⚙️ Machine Learning Pipeline

### 1️⃣ Data Preprocessing

* Data Loading
* Feature Selection
* Variance Threshold Filtering
* Standard Scaling
* Target (RUL) Generation

### 2️⃣ Model Training

Two models were evaluated:

* Random Forest Regressor
* XGBoost Regressor

### 3️⃣ Model Evaluation

Metrics Used:

* MAE (Mean Absolute Error)
* RMSE (Root Mean Squared Error)
* R² Score

---

## 📈 Model Performance

| Model         | MAE   | RMSE  | R² Score |
| ------------- | ----- | ----- | -------- |
| Random Forest | 13.21 | 18.14 | 0.81     |
| XGBoost       | 12.97 | 17.78 | 0.82     |

### 🏆 Best Model

**XGBoost Regressor**

Reason:

* Lowest Prediction Error
* Highest R² Score
* Better Generalization Performance

Final Model:

```text
XGBoost Regressor
R² Score = 0.82
```

---

## 🔍 Feature Importance Analysis

The model identifies the most influential engine sensors affecting Remaining Useful Life prediction.

Benefits:

* Better interpretability
* Identification of critical degradation indicators
* Improved maintenance planning

---

## 🖥️ Streamlit Dashboard

The project includes a fully interactive Streamlit application.

### Features

✅ Upload Test Dataset

✅ Select Engine ID

✅ Real-time RUL Prediction

✅ Manual Sensor Input

✅ What-If Analysis

✅ Engine Health Monitoring

✅ Feature Importance Visualization

---

## 🟢 Engine Health Classification

The dashboard categorizes engine condition based on predicted RUL.

| Predicted RUL   | Health Status           |
| --------------- | ----------------------- |
| > 100 Cycles    | 🟢 Healthy              |
| 50 – 100 Cycles | 🟡 Monitor              |
| < 50 Cycles     | 🔴 Maintenance Required |

---

## 🛠️ Technologies Used

### Programming

* Python

### Data Processing

* Pandas
* NumPy

### Machine Learning

* Scikit-Learn
* XGBoost

### Visualization

* Matplotlib

### Deployment

* Streamlit

---

## 📂 Project Structure

```text
Turbofan-RUL-Prediction/
│
├── app.py
├── requirements.txt
├── xgboost_rul_model.pkl
├── scaler.pkl
├── selector.pkl
│
├── dataset/
│   ├── train_FD001.txt
│   ├── test_FD001.txt
│   └── RUL_FD001.txt
│
├── screenshots/
│
├── Turbofan_RUL_XGBoost_Improved.ipynb
│
└── README.md
```

---

## ▶️ Installation

Clone Repository

```bash
git clone https://github.com/03-princy/Turbofan-RUL-Prediction.git
```

Move into project folder

```bash
cd Turbofan-RUL-Prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run Streamlit App

```bash
streamlit run app.py
```

---

## 📷 Screenshots

### Dashboard



### Prediction Result

(Add Screenshot Here)

### Model Comparison

(Add Screenshot Here)

---

## 💡 Key Learnings

* Predictive Maintenance
* Regression Modeling
* Feature Engineering
* XGBoost Optimization
* Model Evaluation
* Streamlit Deployment
* Industrial AI Applications

---

## 🔮 Future Improvements

* Deep Learning (LSTM) Based RUL Prediction
* Real-Time Sensor Streaming
* Cloud Deployment
* Multi-Dataset Support
* Advanced Explainable AI (SHAP)

---

## 👩‍💻 Author

**Priyanka Singh**

MCA Student | Machine Learning Enthusiast

Focused on AI, Data Science, Predictive Analytics, and Real-World Machine Learning Applications.

---

## ⭐ If you found this project useful, please give it a star!
