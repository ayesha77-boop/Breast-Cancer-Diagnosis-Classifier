# 🧬 Breast Cancer Diagnosis Classifier

A machine learning pipeline for classifying breast cancer tumours as **Malignant** or **Benign** using the Wisconsin Breast Cancer Dataset. The project covers end-to-end data science: cleaning, exploratory analysis, model building, hyperparameter tuning, and evaluation.

---

## 📁 Project Structure

```
├── Cancer_Data.csv               # Raw dataset
├── ayesha_bioinfo_python_project.py  # Main project script (exported from Colab)
└── README.md
```

---

## 🔍 Workflow Overview

### 1. Data Cleaning
- Loaded the raw CSV dataset
- Removed fully null rows and dropped irrelevant columns (`Unnamed: 32`, `id`)
- Verified shape, dtypes, and null counts post-cleaning

### 2. Exploratory Data Analysis (EDA)
- **Correlation heatmap** across all features
- **Pairplot** across key mean features, coloured by diagnosis
- **Countplot** of Malignant vs. Benign distribution
- **Boxplots** and **Violinplots** for feature comparison by diagnosis
- **Histogram overlays** comparing malignant and benign distributions for 5 core features
- **Mean value comparison table** between tumour classes

### 3. Model Building
Three classifiers were trained and evaluated:

| Model | Description |
|---|---|
| Logistic Regression | Linear baseline with L1/L2 regularisation |
| Random Forest | Ensemble of decision trees with feature importance |
| K-Nearest Neighbors | Distance-based classification |

**Preprocessing steps:**
- Train/test split (80/20, `random_state=42`)
- SMOTE oversampling to handle class imbalance
- Standard scaling of features

### 4. Evaluation Metrics
Each model was assessed using:
- Accuracy, Precision, Recall, F1-Score (for the Malignant class)
- Confusion Matrix (heatmap)
- ROC Curve and AUC Score
- 5-fold Cross-Validation

### 5. Hyperparameter Tuning (GridSearchCV)
- **Logistic Regression:** tuned `C`, `penalty`, `solver`
- **Random Forest:** tuned `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`
- **KNN:** tuned `n_neighbors`, `weights`

---

## 📊 Results Summary

| Model | Pre-Tuning Accuracy | Post-Tuning Accuracy |
|---|---|---|
| Logistic Regression | — | **98.25%** ✅ |
| Random Forest | — | ~97% |
| K-Nearest Neighbors | — | 93.86% |

Logistic Regression achieved the highest accuracy post-tuning at **98.25%**, while KNN performed the lowest at **93.86%**.

---

## 🛠️ Tech Stack

- **Python 3**
- `pandas`, `numpy` — data manipulation
- `matplotlib`, `seaborn` — visualisation
- `scikit-learn` — modelling, evaluation, tuning
- `imbalanced-learn (imblearn)` — SMOTE oversampling
- **Google Colab** — development environment

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/your-username/breast-cancer-diagnosis-classifier.git
cd breast-cancer-diagnosis-classifier

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn

# Run the script
python ayesha_bioinfo_python_project.py
```

> **Note:** Place `Cancer_Data.csv` in the root directory (or update the file path in the script) before running.

---

## 📌 Dataset

The project uses the **Wisconsin Breast Cancer Diagnostic Dataset**, which contains 30 numeric features computed from digitised images of fine needle aspirates (FNA) of breast masses.

- **Target variable:** `diagnosis` — M (Malignant) or B (Benign)
- **Features:** radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, and fractal dimension (mean, SE, and worst values)

---

## 👩‍💻 Author

**Ayesha**
Bioinformatics Python Project
