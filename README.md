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
