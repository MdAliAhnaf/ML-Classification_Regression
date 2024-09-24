# Traditional Machine Learning Classification and Regression

This repository contains two machine learning tasks: a regression task and a classification task, implemented in separate Jupyter notebooks within their respective folders.

## Folder Structure

/ML-Classification_Regression ├── traditional_ml_regression │ ├── ensemble_predictions_XGB_rf_LGBM_DATA-AUG.csv │ ├── ensemble_predictions_XGB_rf_LGBM_DATA-AUG_k-Fold.csv │ ├── lgbm_predictions.csv │ ├── MD._ALI_AHNAF.csv │ ├── ml_regression.ipynb │ ├── test.csv │ └── train.csv └── traditional_ml_classification ├── decision_tree_plot.png ├── ml_classification.ipynb ├── predicted_labels.csv ├── predicted_testX_labels.csv ├── test.csv └── train.csv


## 1. Regression Task (`ml_regression.ipynb`)

### Dataset Overview
The regression task utilizes a dataset with the following structure:

**Training Data (`train.csv`):**
- **Columns:**
    - `Id`: Unique identifier for each record (1 to 139652)
    - `col4`: Continuous feature
    - `col6`: Continuous feature
    - `col9`: Categorical feature
    - `col1`: Date feature
    - `col8`: Continuous feature
    - `col3`: Categorical feature
    - `col7`: Continuous feature
    - `y`: Target variable (continuous)

- **Example of the first row:**
    ```
    1, 2107, 5, B0, 3/10/2014, 3, A0, 2, 1300000
    ```

**Test Data (`test.csv`):**
- **Columns:**
    - `Id`: Unique identifier for each record (139653 to 199504)
    - Other features similar to the training data but without the target variable `y`.

### Methodology
1. **Data Preprocessing:**
    - Encoding categorical variables.
    - Computing correlation coefficients between input columns and the target variable.
    - Combining training and test data for label encoding.
    - Standardizing continuous features to ensure uniformity in scales.

2. **Model Training:**
    - Employed models: `LGBMRegressor` and `XGBRegressor`.
    - Utilized Grid Search for hyperparameter tuning:
        - **LGBMRegressor Best Hyperparameters:**
            - `max_depth`: 5
            - `n_estimators`: 300
            - `num_leaves`: 64
            - `random_state`: 42
        - **XGBRegressor Best Hyperparameters:**
            - [Include specific parameters once available]
    
3. **Model Evaluation:**
    - Used RMSE as the evaluation metric.
    - Achieved the **highest public score**: **5671017705258.910156** on Kaggle Competition.

### Results
| Model              | Best Hyperparameters                                     | Public Score                       |
|--------------------|---------------------------------------------------------|------------------------------------|
| LGBMRegressor      | max_depth=5, n_estimators=300, num_leaves=64          | 5671017705258.910156               |
| XGBRegressor       | [Include best hyperparameters here]                     | 5671017705258.910156               |

---

## 2. Classification Task (`ml_classification.ipynb`)

### Dataset Overview
The classification task utilizes a dataset structured as follows:

**Training Data (`train.csv`):**
- **Columns:**
    - `ID`: Unique identifier for each record (1 to 6000)
    - `col_0`: Categorical feature
    - `col_1`: Continuous feature
    - `col_2`: Continuous feature
    - `col_3`: Continuous feature
    - `col_4`: Continuous feature
    - `col_5`: Continuous feature
    - `y`: Target variable (categorical)

- **Example of the first row:**
    ```
    1, A1, 0.423913043, 310.7, -1.302803264, 0.889328063, 1737, C3
    ```

**Test Data (`test.csv`):**
- **Columns:**
    - `ID`: Unique identifier for each record (6001 to 10000)
    - Other features similar to the training data but without the target variable `y`.

### Methodology
1. **Data Preprocessing:**
    - Encoding categorical variables using Label Encoding.
    - Computing correlation coefficients between input columns and the target variable.
    - Combining training and test data for label encoding.
    - Splitting the combined data back into `X_train` and `X_test`.
    - Converting categorical labels to numeric values for classification.

2. **Model Training:**
    - Implemented `Stratified K-Fold` cross-validation to ensure balanced representation of classes.
    - Initialized classifiers, including:
        - `RandomForestClassifier`
        - `XGBoost`
        - `Logistic Regression`
        - `Gradient Boosting`
        - `Naive Bayes`
        - `LGBMClassifier`
        - `CatBoostClassifier`
    - Combined predictions using a `Voting Classifier` with soft voting.
    - Addressed class imbalance using `SMOTE` to oversample the minority class for each fold.

3. **Model Evaluation:**
    - Evaluated classifier performance based on accuracy scores.
    - Generated a classification report detailing precision, recall, and F1-score.

### Results
#### Classification Report
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.99      | 0.98   | 0.99     | 1158    |
| 1     | 1.00      | 1.00   | 1.00     | 1158    |
| 2     | 0.99      | 1.00   | 0.99     | 1158    |
| 3     | 0.99      | 1.00   | 1.00     | 1159    |
| 4     | 1.00      | 0.99   | 1.00     | 1159    |
| 5     | 1.00      | 1.00   | 1.00     | 1158    |
| **Accuracy** | **1.00** | -      | -        | 6950    |
| **Macro Avg** | **1.00** | **1.00** | **1.00** | 6950    |
| **Weighted Avg** | **1.00** | **1.00** | **1.00** | 6950    |

### Best Classifier
- For the classification task, **XGBoost** was selected as the best classifier, and the classification report was generated solely based on its performance.

### Predictions
- Generated predictions on the test data and saved them as a CSV file (`predicted_labels.csv`).
- Converted encoded labels back to original categorical labels.

---
