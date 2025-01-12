# CS-412-Term-Project

# 1- Instagram Profile Data Analysis and Classification

## Overview

This project categorizes Instagram profiles based on textual data derived from user bios, captions, and metadata. It leverages machine learning techniques for feature extraction, model training, and prediction, providing a robust solution for social media analytics.

## Project Structure

The project is organized across three main notebooks:

1. **`processing.ipynb`**
   - Loads and preprocesses raw Instagram data.
   - Cleans and tokenizes text, removes stopwords, and performs lemmatization.
   - Outputs cleaned data for further analysis.

2. **`models.ipynb`**
   - Converts text data to numerical vectors using TF-IDF.
   - Trains and compares classification models, including Logistic Regression, SVM, and Random Forest.
   - Utilizes SMOTE to address class imbalances and optimize model performance.

3. **`selected_model.ipynb`**
   - Focuses on the best-performing model: Logistic Regression with One-vs-Rest (OvR).
   - Fine-tunes SMOTE thresholds and retrains the model.
   - Saves the trained model and vectorizer for future use.
   - Demonstrates predictions on unseen data.

## Tools and Libraries

- **Python Libraries**: `pandas`, `NumPy`, `scikit-learn`, `imblearn`, `joblib`
- **Feature Extraction**: TF-IDF Vectorizer
- **Environment**: Jupyter Notebook

## Key Features

- **Preprocessing**: Comprehensive cleaning and preparation of textual data.
- **Class Imbalance Handling**: SMOTE ensures balanced datasets for improved model generalization.
- **Best Model**: Logistic Regression with OvR provides high accuracy and balanced predictions across categories.

## Outputs

- **Cleaned Data**: `processed_profiles.json`
- **Trained Model**: `logistic_regression_ovr_smote_600.pkl`
- **Predictions**: `prediction-classification-round*.json`

## Conclusion

This pipeline efficiently processes and classifies Instagram profiles into predefined categories. By addressing challenges like class imbalance and using well-optimized machine learning models, the project demonstrates an effective approach to social media analytics.

---

# 2- Regression Pipeline for Predicting Post Engagement

## Overview
This project implements a robust regression pipeline to predict the `like_count` of social media posts. It incorporates advanced machine learning models, feature engineering, and hyperparameter tuning to ensure high accuracy and reliability. The solution is tailored for engagement analytics, providing insights into post performance.

## Project Structure
The project is structured into distinct modules to streamline the workflow:

### Data Loading and Preprocessing
- **Objective:** Transforms raw JSON data into a structured format suitable for machine learning.
- **Key Steps:**
  - Extracts user and post-level features like `follower_count`, `media_type`, and `time_index`.
  - Engineers features such as `follower_following_ratio` and caps outliers in `like_count` at the 99th percentile.
  - Produces a cleaned and feature-engineered DataFrame.

### Model Training and Evaluation
- **Objective:** Identifies the best-performing regressor through multi-model comparison.
- **Key Steps:**
  - Evaluates Random Forest, XGBoost, LightGBM, and CatBoost models.
  - Tunes hyperparameters using `RandomizedSearchCV` with cross-validation.
  - Selects the best model based on Mean Squared Error (MSE) and R² Score on test data.

### Prediction Workflow
- **Objective:** Predicts `like_count` for unseen test data.
- **Key Steps:**
  - Loads the trained model from a serialized file (`best_model.pkl`).
  - Prepares test data by encoding features and ensuring consistency with training data.
  - Outputs predictions in a JSON file (`prediction-regression-round3.json`).

## Tools and Libraries
- **Programming Language:** Python
- **Libraries:** pandas, NumPy, scikit-learn, xgboost, lightgbm, catboost
- **Workflow Management:** Jupyter Notebooks and Python scripts
- **File I/O:** JSON and pickle formats

## Key Features
1. **Feature Engineering:**
   - Introduces `follower_following_ratio` to capture user-level dynamics.
   - Applies one-hot encoding to categorical features for improved model compatibility.

2. **Multi-Model Comparison:**
   - Evaluates and tunes four advanced models to ensure optimal performance.

3. **Prediction Reliability:**
   - Ensures robust predictions through consistent preprocessing of test data.

## Outputs
- **Cleaned Data:** Structured DataFrame ready for training and testing.
- **Best Model:** The top-performing regressor with optimized hyperparameters.
- **Predictions:** JSON file mapping `post_id` to predicted `like_count`.

## Conclusion
This pipeline demonstrates a comprehensive approach to social media analytics, combining advanced regression techniques and rigorous preprocessing. By addressing challenges like outliers and class imbalance, the project offers a reliable framework for predicting post engagement.

## Future Work
- Integrate additional features such as sentiment analysis of captions.
- Explore ensemble methods for combining multiple models.
- Optimize runtime performance for large-scale datasets.
