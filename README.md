# 🔍 Credit Card Fraud Detection Case Study

This project aims to detect fraudulent credit card transactions using machine learning techniques. 
We built and compared multiple models to identify fraud effectively in a highly imbalanced dataset. 
This is part of a case study to simulate a real-world fraud detection scenario.

---

## 📁 Dataset Overview

The dataset contains anonymized credit card transaction records with the following key features:

- `amount`: Transaction amount
- `category`: Type of transaction (e.g., gas, food, etc.)
- `gender`, `age`: Customer demographic information
- `step`: Time step (1 step = 1 hour)
- `fraud`: Target variable (1 = fraud, 0 = non-fraud)

Additional features were engineered, including:
- `hour_of_day`
- `time_period` (morning, afternoon, evening, night)
- `is_high_amount`

---

## 🛠️ Project Workflow

1. **Data Cleaning & Preprocessing**
   - Removed unnecessary columns (e.g., customer ID, merchant codes).
   - Created new features: `hour_of_day`, `time_period`, and `is_high_amount`.

2. **Exploratory Data Analysis (EDA)**
   - Analyzed class imbalance.
   - Examined fraud patterns across different time periods.

3. **Feature Encoding**
   - Applied `StandardScaler` to numeric features.
   - Applied `OneHotEncoder` to categorical features via a `ColumnTransformer`.

4. **Handling Imbalance**
   - Applied **SMOTE** to oversample the minority class (fraud cases) and balance the training data.

5. **Train/Test Split**
   - 80% training, 20% testing with stratification on the `fraud` label.

---

## 🤖 Models Used & Evaluation

We implemented and compared the following machine learning models:

### 1. **Logistic Regression**
- Used as a baseline model for interpretability.
- Trained with:
  - `class_weight='balanced'`
  - `solver='liblinear'`

### 2. **Random Forest Classifier**
- Ensemble tree-based model.
- Good at capturing nonlinear relationships and feature interactions.
- Trained with:
  - `class_weight='balanced'`
  - **Tuned using `GridSearchCV`** with parameters:
    ```python
    param_grid = {
        'n_estimators': [100],
        'max_depth': [None, 10],
        'min_samples_split': [2]
    }
    ```
  - `cv=2` used for cross-validation due to memory constraints.

### 3. **Gradient Boosting Classifier**
- Boosting technique that corrects errors iteratively.
- Naturally handles class imbalance better.
- Showed good precision and ROC AUC results.

---

### 🧪 Evaluation Metrics

All models were evaluated on the **test set** using:

| Metric                    | Description                                                        |
|---------------------------|--------------------------------------------------------------------|
| **F1-score**              | Balances precision and recall; ideal for imbalanced data.          |
| **ROC AUC**               | Measures model's ability to distinguish fraud from non-fraud.      |
| **Confusion Matrix**      | Shows true/false positives and negatives.                          |
| **Classification Report** | Detailed per-class metrics: precision, recall, F1-score.           |

---

## 📊 Key Outcomes & Recommendations

- ✅ **Random Forest and Gradient Boosting** outperformed Logistic Regression, especially in F1-score.
- ✅ **SMOTE** significantly improved the model's ability to detect fraud (higher recall).
- ✅ Features such as:
  - `amount`
  - `is_high_amount`
  - `time_period`  
  were highly indicative of fraudulent behavior.
- 🔁 We recommend focusing on these features and models when building fraud detection systems in production.

------------
