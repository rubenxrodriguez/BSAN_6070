# BSAN6070_TW07
# PCA and XGBoost Model Evaluation on Breast Cancer Dataset

## Overview
This project demonstrates the application of Principal Component Analysis (PCA) for dimensionality reduction and the use of XGBoost for classification on the Breast Cancer dataset from `sklearn.datasets`. It also evaluates feature importance using SHAP and Gain metrics and tests model stability by introducing noise and shuffling data. The motivation behind using PCA to improve the accuracy and stability of XGBoost models comes from the research article : https://medium.com/georgian-impact-blog/on-the-trustworthiness-of-tree-ensemble-methods-ce62df5d1482

## Dependencies
Ensure you have the following Python libraries installed before running the script:

```bash
pip install numpy pandas scikit-learn xgboost shap matplotlib scipy
```

## Data
The script uses the Breast Cancer dataset from `sklearn.datasets`, which includes:
- Features: 30 numerical attributes related to cell nuclei characteristics.
- Target: Binary classification (Malignant = 1, Benign = 0).

## Workflow

### 1. Data Preprocessing
- Load the dataset and split it into training and testing sets.
- Standardize the features using `StandardScaler`.

### 2. PCA Dimensionality Reduction
- Perform PCA with `n_components=2` for visualization.
- Perform PCA with `n_components=6` to retain most of the variance.
- Display explained variance ratios.
- Visualize PCA results using a scatter plot.

### 3. Model Training and Evaluation
- Train an `XGBClassifier` on PCA-reduced data and original scaled data.
- Compare test accuracy between PCA-transformed features and original features.

### 4. Feature Importance Analysis
- Compute Gain-based feature importance for both PCA components and original features.
- Use SHAP to analyze feature impact on predictions.
- Visualize feature importance using bar charts and SHAP summary plots.

### 5. Model Stability Testing
- Introduce noise and shuffle data for robustness testing.
- Retrain models multiple times and assess stability.
- Compute and compare SHAP and Gain stability scores for PCA components and original features.
- Use Spearman correlation to quantify stability across iterations.

## Results
- The script compares the effectiveness of dimensionality reduction via PCA against training on full-featured data.
- SHAP and Gain feature importance metrics reveal the most influential features/components.
- Stability testing evaluates the robustness of feature importance rankings.

## How to Run
Execute the script in a Python environment:
```bash
python script.py
```
Ensure all dependencies are installed before execution.

## Visualization Outputs
The script generates:
- PCA scatter plot
- Gain importance bar charts
- SHAP summary plots

## Conclusion
This project highlights:
- The trade-offs between PCA-based feature reduction and full-featured training.
- The impact of PCA on model interpretability.
- The importance of stability testing in feature selection.
