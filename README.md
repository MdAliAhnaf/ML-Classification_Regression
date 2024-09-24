# Traditional Machine Learning (Classification and Regression)

This repository contains two machine learning tasks: a regression task and a classification task, implemented in separate Jupyter notebooks within their respective folders.

## Folder Structure

/ML-Classification_Regression 
    ── traditional_ml_regression
        └── .ipynb_checkpoints
            └── ml_regression-checkpoint.ipynb
        └── ensemble_predictions_XGB_rf_LGBM_DATA-AUG.csv
        └── ensemble_predictions_XGB_rf_LGBM_DATA-AUG_k-Fold.csv
        └── lgbm_predictions.csv
        └── MD._ALI_AHNAF.csv
        └── ml_regression.ipynb
        └── test.csv
        └── train.csv
        └── xgb_predictions.csv

    ── traditional_ml_classification
        └── .ipynb_checkpoints
            └── ml_classification-checkpoint.ipynb
        └── decision_tree_plot.png
        └── ml_classification.ipynb
        └── predicted_labels.csv
        └── predicted_testX_labels.csv
        └── test.csv
        └── train.csv

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
1. **Loading Data**: Datasets `train.csv` and `test.csv` are loaded into the notebook.

2. **Data Preprocessing:**
    - **Encoding Categorical Variables**: Categorical features are encoded to numerical values for model compatibility.
    - **Computing Correlation Coefficients**: Correlation coefficients between input features and the target variable are computed to assess feature importance.
    - **Combining Data for Label Encoding**: The training and test datasets are combined to ensure consistent label encoding across both datasets.
    - **Handling Categorical Variables**: Label Encoding is applied to convert categorical features into numeric values.
    - **Separating Data**: After label encoding, the combined dataset is split back into `X_train` and `X_test`.
    - **Standardizing Continuous Features**: Continuous numerical features are standardized to improve model performance.

3. **Stratified K-Fold**: Stratified K-Fold cross-validation is applied to ensure that the training and validation datasets maintain the same distribution of the target variable.

4. **Model Training:**
    - Employed models: `LGBMRegressor` and `XGBRegressor`.
    - Utilized Grid Search for hyperparameter tuning:
        - **LGBMRegressor Best Hyperparameters:**
            - `max_depth`: 5
            - `n_estimators`: 300
            - `num_leaves`: 64
            - `random_state`: 42
        - **XGBRegressor Best Hyperparameters:**
            - *[Grid Search was used to find the best hyperparameters]*
                - `max_depth`: [3, 4, 5]
                - `n_estimators`: [100, 200, 300]
                - `learning_rate`: [0.1, 0.01, 0.001]
                - `random_state`: 42

5. **Data Augmentation**: Data augmentation techniques were employed to improve the generalization of the models.

6. **Regularization & Learning Rate Scheduling**: Regularization techniques and learning rate scheduling were used to prevent overfitting and optimize training.

7. **Early Stopping**: Early stopping was utilized during training to prevent overfitting by halting training when performance stopped improving.

8. **Creating Predictions**: Predictions on the test dataset are generated and saved in a DataFrame with corresponding IDs.

9. **Preparing for Submission**: Final predictions are prepared and saved in a `.csv` file for submission to Kaggle.

10. **Model Evaluation:**
    - Used Root Mean Squared Error (RMSE) as the evaluation metric.
    - **Best Models**:
  - `LGBMRegressor` and `XGBRegressor` achieved the **highest public score of 5671017705258.910156** on the Kaggle competition.

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

1. **Loading Data**: Datasets `train.csv` and `test.csv` are loaded into the notebook.

2. **Data Preprocessing:**
    - **Encoding Categorical Variables**: Categorical features are encoded using Label Encoding for compatibility with the models.
    - **Computing Correlation Coefficients**: Correlation coefficients between input features and the target variable are computed for feature selection and analysis.
    - **Combining Data for Label Encoding**: The training and test datasets are combined to ensure consistent encoding across the entire dataset.
    - **Handling Categorical Variables**: Label Encoding is applied to convert categorical features into numeric formats for modeling.
    - **Separating Data**: After label encoding, the combined dataset is split back into `X_train` and `X_test`.
    - **Converting Labels**: Target labels are converted to numeric values to allow classification by the models.

3. **Model Training:**
    - Implemented `Stratified K-Fold` cross-validation to ensure and maintain balanced representation of classes to tackle class imbalance.
    - Initialized classifiers, including:
        - `RandomForestClassifier`
        - `XGBoost`
        - `Logistic Regression`
        - `Gradient Boosting`
        - `Naive Bayes`
        - `LGBMClassifier`
        - `CatBoostClassifier`
    - **Ensemble Classifier**: A Voting Classifier is implemented to combine predictions from multiple classifiers using soft voting for better performance.
    - **Cross-Validation with SMOTE**: Cross-validation is performed using SMOTE to address class imbalance. SMOTE oversamples the minority class in each fold. Addressed class imbalance using `SMOTE` to oversample the minority class for each fold.
    - **Resampling**: The training data is oversampled to create a balanced dataset for final model training.

4. **Model Evaluation:**
    - The classifiers are evaluated on accuracy and other relevant metrics using cross-validation.
    - The best-performing classifier from the cross-validation is retrained on the entire dataset.
    - Generated a classification report detailing precision, recall, and F1-score.

### Results
- **Best Model**: The best classifier was evaluated using a classification report that demonstrated high accuracy across all classes.

### Best Classifier
- For the classification task, **XGBoost** was selected as the best classifier, and the classification report was generated solely based on its performance.

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

### Predictions
- Generated predictions on the test data and saved them as a CSV file with ID and predicted labels(`predicted_labels.csv`).
- Converted encoded labels back to original categorical labels.

---

## Technologies Used

- Python (with libraries such as NumPy, Pandas, Scikit-learn, LightGBM, XGBoost, SMOTE)
- Jupyter Notebooks
- Kaggle API (for submission)

## Evaluation Metrics

- **Regression**: Root Mean Squared Error (RMSE)
- **Classification**: Accuracy, Precision, Recall, F1-score (via Classification Report)

---