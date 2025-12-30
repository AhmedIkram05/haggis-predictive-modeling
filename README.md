# Scottish Haggis Dataset - Data Mining and Predictive Modelling

## 🚀 Overview
This project provides a comprehensive data mining and predictive modelling analysis of the Scottish Haggis dataset. 
It explores the morphological characteristics of different haggis species across various Scottish islands, employing both unsupervised and supervised machine learning techniques to classify species and predict body mass.

## 🧠 Tech Stack
- **Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Visualisation:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-Learn (Clustering, Classification, Regression, PCA)
- **Ensemble Methods:** XGBoost, Random Forest

## 📊 Features
- **Exploratory Data Analysis (EDA):** In-depth analysis of feature distributions, correlations, and species-specific traits.
- **Feature Engineering:** Creation of domain-specific features such as `nose_tail_ratio` and `tail_mass_ratio` to improve model performance.
- **Unsupervised Learning:** Implementation of K-Means (optimal k=3) and DBSCAN clustering, supported by PCA for dimensionality reduction.
- **Supervised Classification:** Comparative analysis of Decision Trees, Random Forest, XGBoost, KNN, and Logistic Regression for species identification.
- **Supervised Regression:** Linear Regression model to predict haggis body mass based on morphological features.

## 📈 Results
- **89.86% Accuracy:** Achieved by Random Forest, XGBoost, KNN, and Logistic Regression in species classification.
- **0.756 R² Score:** The Linear Regression model explains 75.6% of the variance in haggis body mass.
- **Feature Importance:** `tail_length_mm` was consistently identified as the strongest predictor across all modelling stages.

## 🧪 How to Run
1. **Clone the repository:**
   ```bash
   git clone https://github.com/AhmedIkram05/haggis-predictive-modeling.git
   ```
2. **Install dependencies:**
   Ensure you have Python installed, then install the required libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost
   ```
3. **Run the analysis:**
   Open [Ahmed_Ikram_2571642_Final_Project.ipynb](Ahmed_Ikram_2571642_Final_Project.ipynb) in VS Code or Jupyter Notebook and run all cells.
