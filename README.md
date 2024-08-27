# Customer Churn Prediction

This project focuses on predicting customer churn using a variety of machine learning techniques. The goal is to build a model that can accurately identify customers who are at risk of churning, allowing businesses to take proactive measures to retain them.

## Project Overview

Customer churn is a critical issue for businesses that rely on subscription models. This project uses a dataset with various features related to customer behavior and demographics to predict whether a customer will churn.

## Dataset

- **Training Data**: Contains 10,000 entries with 231 features initially, which was reduced to 39 after preprocessing.
- **Testing Data**: Used to validate the model and make final predictions.

## Key Steps in the Workflow

### 1. Data Preprocessing
- **Handling Missing Values**: Removed features with more than 30% missing data.
- **Handling Zero-Inflated Features**: Features with more than 50% zero values were dropped.
- **Categorical Variables**: Encoded using `TargetEncoder` after filling missing values with the mode.
- **Numerical Variables**: Skewed distributions were normalized using `PowerTransformer`.

### 2. Feature Selection
- **High Cardinality Features**: Removed categorical features with more than 300 unique values.
- **High Skew Features**: Features with a skew greater than 5 were dropped as they did not improve model performance even after outlier removal.

### 3. Data Balancing
- **Imbalance Handling**: Used SMOTE (Synthetic Minority Over-sampling Technique) to balance the classes, as the target variable was heavily imbalanced (13% churn rate).

### 4. Model Building and Tuning
- **Model Choice**: LightGBM was chosen after experimenting with various models due to its superior performance.
- **Hyperparameter Tuning**: Optuna was used for hyperparameter optimization, leading to a balanced accuracy score of approximately 96% across cross-validation folds.

### 5. Feature Importance and Interpretation
- **SHAP Values**: Used to identify the most influential features. The most important feature was `Var126`, despite its high proportion of missing values in the original dataset.

## Final Model and Evaluation

The final model was a LightGBM classifier with the following key parameters:

```python
best_params_lgbm = {
    'n_estimators': 312, 
    'learning_rate': 0.1608,
    'num_leaves': 94, 
    'max_depth': 39, 
    'min_child_samples': 78, 
    'min_child_weight': 0.0033, 
    'subsample': 0.5215, 
    'colsample_bytree': 0.7146, 
    'reg_alpha': 0.00033, 
    'reg_lambda': 4.39e-08, 
    'scale_pos_weight': 1.2627
}
```

The model achieved a high balanced accuracy, confirming its effectiveness in predicting customer churn.
## Results

The model predictions are saved as `submission.csv`, which includes the predicted churn status for each customer in the test set.
