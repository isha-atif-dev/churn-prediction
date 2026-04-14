# 🏦 Bank Customer Churn Prediction

A machine learning project that predicts which bank customers are likely to leave — helping banks take action before it's too late.

---

## 📌 What This Project Does

Banks lose millions when customers close their accounts. This project builds a **Random Forest classification model** that analyses customer data and predicts whether a customer will churn (leave) or stay — giving the bank a chance to intervene early.

---

## 📁 Project Structure

```
churn-prediction/
│
├── data/
│   ├── raw/                        # Original dataset — never modified
│   └── processed/                  # Cleaned and split data ready for training
│
├── notebooks/
│   └── 01_eda.ipynb                # Exploratory Data Analysis
│
├── src/
│   ├── preprocess.py               # Data cleaning, encoding, train-test split
│   ├── train.py                    # Model training and saving
│   └── evaluate.py                 # Model evaluation and metrics
│
├── models/
│   └── churn_model.pkl             # Saved trained model
│
├── requirements.txt                
└── README.md                       
```

---

## 🔍 Dataset

**Source:** [Churn Modelling Dataset — Kaggle](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling)

- 10,000 bank customers
- 14 features including age, balance, credit score, geography, activity status
- Target variable: `Exited` (1 = churned, 0 = stayed)

---

## 🧠 Key Findings from EDA

- **80% of customers stayed, 20% churned** — class imbalance present
- **Customers aged 40+** showed significantly higher churn rates
- **Germany** had a disproportionately high churn rate vs France and Spain
- **Balance = 0** is a strong signal of an inactive, at-risk customer

---

## ⚙️ How It Works

**Step 1 — Preprocessing (`preprocess.py`)**
- Dropped irrelevant columns: `RowNumber`, `CustomerId`, `Surname`
- Applied One Hot Encoding to `Geography` and `Gender`
- Split data: 80% training, 20% testing

**Step 2 — Training (`train.py`)**
- Model: `RandomForestClassifier` with `class_weight='balanced'`
- Trained on 8,000 customers
- Model saved to `models/churn_model.pkl`

**Step 3 — Evaluation (`evaluate.py`)**
- Tested on 2,000 unseen customers

---

## 📊 Results

| Metric | Score |
|--------|-------|
| Accuracy | 86.95% |
| Precision | 77.73% |
| Recall | 47.07% |
| F1 Score | 58.63% |

> **Note:** Accuracy alone is misleading due to class imbalance. Recall is the most important metric here — it measures how many real churners the model actually caught.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| Pandas | Data manipulation |
| Scikit-learn | ML model and metrics |
| Matplotlib / Seaborn | Data visualisation |
| Joblib | Model saving |
| Jupyter Notebook | EDA and exploration |

---

## 🚀 How to Run

**1. Clone the repository**
```bash
git clone https://github.com/isha-atif-dev/churn-prediction.git
cd churn-prediction
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run preprocessing**
```bash
python src/preprocess.py
```

**4. Train the model**
```bash
python src/train.py
```

**5. Evaluate the model**
```bash
python src/evaluate.py
```

---

## 💡 What I Learned

- How to structure a professional ML project from scratch
- Exploratory Data Analysis — letting data tell the story before modelling
- One Hot Encoding for categorical variables
- Train-test split and why it matters
- Random Forest — ensemble learning with hundreds of decision trees
- Why accuracy alone is misleading (class imbalance problem)
- Precision, Recall, and F1 Score — and when each one matters
- Saving and loading models with joblib
- Version controlling an ML project with Git and GitHub

---

## 👩‍💻 Author

**Isha Atif**  
MRes Applied Artificial Intelligence Student  
[GitHub](https://github.com/isha-atif-dev) • [LinkedIn](https://linkedin.com/in/isha-atif)