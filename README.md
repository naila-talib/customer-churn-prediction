# 📊 Customer Churn Prediction AI

An end-to-end Machine Learning project that predicts whether a customer is likely to churn or stay based on customer demographics, services, contracts, usage, and billing information.

## 🚀 Project Overview

Customer churn is a major challenge for subscription-based businesses. This project uses Machine Learning to identify customers who are at risk of leaving and provides churn probability and risk level through a Flask web application.

### Key Features

- Customer churn prediction
- Churn probability
- LOW, MEDIUM, and HIGH risk levels
- Logistic Regression baseline
- Random Forest classification
- Model comparison and evaluation
- Confusion Matrix
- ROC Curve
- Feature Importance
- Interactive Flask web application

## 🎯 Objectives

1. Analyze customer churn data.
2. Handle missing values.
3. Process numerical and categorical features.
4. Train multiple classification models.
5. Compare model performance.
6. Select the best model.
7. Evaluate the model using multiple metrics.
8. Save the trained model.
9. Deploy the model with Flask.

## 🧠 Machine Learning Problem

This is a Binary Classification problem.

Target variable:

`Customer Status`

Predictions:

- Stayed → 0
- Churned → 1

The `Joined` category is excluded because it represents newly joined customers rather than a final churn/stay outcome.

## 📂 Dataset

Dataset size:

- Rows: 7043
- Columns: 38

Customer Status distribution:

- Stayed: 4720
- Churned: 1869
- Joined: 454

Records used for modeling:

- 6589

The dataset contains customer demographics, services, usage, contracts, billing information, and customer status.

## 📋 Main Features

### Customer Information

- Customer ID
- Gender
- Age
- Married
- Number of Dependents
- City
- Zip Code
- Latitude
- Longitude
- Number of Referrals

### Service Information

- Phone Service
- Multiple Lines
- Internet Service
- Internet Type
- Online Security
- Online Backup
- Device Protection Plan
- Premium Tech Support
- Streaming TV
- Streaming Movies
- Streaming Music
- Unlimited Data

### Usage and Billing

- Tenure in Months
- Avg Monthly GB Download
- Avg Monthly Long Distance Charges
- Contract
- Paperless Billing
- Payment Method
- Monthly Charge
- Total Charges
- Total Refunds
- Total Extra Data Charges
- Total Long Distance Charges
- Total Revenue

## ⚠️ Data Leakage Prevention

The following fields are excluded from model training:

- Customer ID
- Customer Status
- Churn Category
- Churn Reason

These fields can directly reveal customer outcomes and could cause data leakage.

## 🏗️ Project Workflow

```text
Customer Dataset
       ↓
Exploratory Data Analysis
       ↓
Data Cleaning
       ↓
Feature Selection
       ↓
Data Preprocessing
       ↓
Train/Test Split
       ↓
Machine Learning Models
       ↓
Model Comparison
       ↓
Best Model Selection
       ↓
Model Evaluation
       ↓
Saved Model
       ↓
Flask Web Application
       ↓
Customer Input
       ↓
Churn Prediction
       ↓
Probability + Risk Level
```

## 🤖 Machine Learning Models

### Logistic Regression

Performance:

- Accuracy: 82.17%
- Precision: 64.33%
- Recall: 83.42%
- F1 Score: 72.64%
- ROC-AUC: 90.62%

### Random Forest

Performance:

- Accuracy: 83.99%
- Precision: 68.91%
- Recall: 79.41%
- F1 Score: 73.79%
- ROC-AUC: 92.08%

Random Forest was selected as the final model.

## 🏆 Final Model

**Best Model: Random Forest**

| Metric | Score |
|---|---:|
| Accuracy | 83.99% |
| Precision | 68.91% |
| Recall | 79.41% |
| F1 Score | 73.79% |
| ROC-AUC | 92.08% |

Training samples: **5271**

Testing samples: **1318**

## ⚙️ Data Preprocessing

### Numerical Features

- Missing-value imputation
- Median replacement
- StandardScaler

Numerical features: **15**

### Categorical Features

- Missing-value imputation
- Most-frequent replacement
- One-Hot Encoding

Categorical features: **19**

## 📊 Model Evaluation

The project evaluates models using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix
- ROC Curve
- Feature Importance

## 📈 Generated Graphs

The evaluation script generates:

- Churn Distribution
- Confusion Matrix
- ROC Curve
- Feature Importance

Graphs are saved in:

```text
static/graphs/
```

## 🌐 Flask Web Application

The Flask application provides an interactive customer information form.

The user enters customer details and receives:

- Churn prediction
- Churn probability
- Risk level
- Model performance
- Model analysis graphs

Example:

```text
Customer is likely to STAY
Churn Probability: 38%
Risk Level: LOW
```

## 🚦 Risk Levels

The application currently uses:

```text
Below 40%       → LOW
40% - 69.99%    → MEDIUM
70% or higher   → HIGH
```

These thresholds can be changed according to business requirements.

## 💼 Business Use Case

Businesses can use churn prediction to identify customers who may leave and take preventive actions such as:

- Special discounts
- Personalized offers
- Loyalty programs
- Customer support
- Better service plans
- Contract upgrades
- Retention campaigns

Machine Learning predictions should support business decisions rather than replace human judgment.

## 📁 Project Structure

```text
09-customer-churn-prediction/
│
├── data/
│   └── customer_churn.csv
│
├── models/
│   ├── churn_model.pkl
│   ├── features.json
│   └── metrics.json
│
├── static/
│   ├── style.css
│   └── graphs/
│       ├── churn_distribution.png
│       ├── confusion_matrix.png
│       ├── roc_curve.png
│       └── feature_importance.png
│
├── templates/
│   └── index.html
│
├── app.py
├── train.py
├── evaluation.py
├── eda.py
├── requirements.txt
├── .gitignore
└── README.md
```

## 📄 File Descriptions

### `eda.py`

Performs exploratory data analysis, including dataset shape, columns, missing values, and customer status distribution.

### `train.py`

Loads and preprocesses the dataset, removes data leakage, trains Logistic Regression and Random Forest models, compares them, selects the best model, and saves the model files.

### `evaluation.py`

Loads the trained model, calculates evaluation metrics, and generates the project graphs.

### `app.py`

Runs the Flask web application, receives customer information, generates predictions, calculates churn probability, and assigns a risk level.

### `templates/index.html`

Contains the web application's HTML interface.

### `static/style.css`

Contains the application's CSS styling.

### `models/churn_model.pkl`

Saved Random Forest Machine Learning model.

### `models/features.json`

Stores feature information required by the application.

### `models/metrics.json`

Stores model evaluation metrics.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib
- Flask
- HTML
- CSS
- Visual Studio Code

## 💻 Installation

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd 09-customer-churn-prediction
```

### 2. Create Virtual Environment

Windows:

```bash
python -m venv venv
```

Activate:

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## ▶️ Run the Project

### Step 1: Run EDA

```bash
python eda.py
```

### Step 2: Train the Model

```bash
python train.py
```

This creates:

```text
models/churn_model.pkl
models/features.json
models/metrics.json
```

### Step 3: Evaluate the Model

```bash
python evaluation.py
```

This creates:

```text
static/graphs/churn_distribution.png
static/graphs/confusion_matrix.png
static/graphs/roc_curve.png
static/graphs/feature_importance.png
```

### Step 4: Start Flask

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5050
```

## 📊 Final Results

The final Random Forest model achieved:

```text
Accuracy  : 83.99%
Precision : 68.91%
Recall    : 79.41%
F1 Score  : 73.79%
ROC-AUC   : 92.08%
```

## 🚀 Future Improvements

Possible improvements include:

- Hyperparameter tuning
- Cross-validation
- XGBoost
- LightGBM
- Neural Networks
- SHAP explainability
- Customer segmentation
- Advanced feature engineering
- Model monitoring
- REST API
- Docker deployment
- Cloud deployment
- Database integration
- Customer history dashboard
- Automated retention recommendations

## 🎓 Learning Outcomes

This project demonstrates practical skills in:

- Python
- Pandas
- NumPy
- Data Cleaning
- Exploratory Data Analysis
- Missing Value Handling
- Feature Selection
- Feature Scaling
- One-Hot Encoding
- Binary Classification
- Logistic Regression
- Random Forest
- Model Comparison
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix
- ROC Curve
- Feature Importance
- Model Serialization
- Flask
- HTML
- CSS
- Machine Learning Deployment

## 📌 Project Highlights

- End-to-End Machine Learning Project
- Customer Churn Prediction
- Binary Classification
- Data Leakage Prevention
- Missing Value Handling
- Numerical Feature Processing
- Categorical Feature Processing
- Logistic Regression
- Random Forest
- Model Comparison
- Best Model Selection
- 83.99% Accuracy
- 92.08% ROC-AUC
- Churn Probability
- Customer Risk Level
- Confusion Matrix
- ROC Curve
- Feature Importance
- Flask Web Application
- Real-Time Prediction Interface

## 👩‍💻 Author

### Naila Usman

AI/ML Student & Developer

Areas of interest:

- Artificial Intelligence
- Machine Learning
- Generative AI
- Python
- Data Science
- AI Web Applications

## ⭐ Portfolio Project

This project is suitable for:

- GitHub Portfolio
- LinkedIn Portfolio
- AI/ML Course Project
- Machine Learning Demonstration
- Client Demonstration
- Practical Machine Learning Application

## 📜 License

This project is created for educational and portfolio purposes.

You are free to modify and improve the project for learning and demonstration.

## ❤️ Project Summary

This Customer Churn Prediction project demonstrates the complete Machine Learning lifecycle:

```text
Dataset
   ↓
EDA
   ↓
Data Cleaning
   ↓
Feature Selection
   ↓
Preprocessing
   ↓
Model Training
   ↓
Model Comparison
   ↓
Best Model Selection
   ↓
Model Evaluation
   ↓
Model Saving
   ↓
Flask Deployment
   ↓
Customer Input
   ↓
Churn Prediction
   ↓
Probability
   ↓
Risk Level
```

The project combines Machine Learning, Data Science, Python, and Flask to create a practical end-to-end AI application for predicting customer churn.
