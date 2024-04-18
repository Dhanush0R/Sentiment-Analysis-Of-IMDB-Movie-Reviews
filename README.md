---

# Sentiment Analysis Of IMDB Movie Reviews

## Introduction
This project focuses on sentiment analysis of IMDB movie reviews using machine learning techniques. The goal is to build a model that can classify movie reviews as either positive or negative based on their content.

## Dataset
The dataset used for this project is the "IMDB Movie Reviews Dataset" available on Kaggle. It contains 50,000 movie reviews along with their corresponding sentiments (positive or negative).

## Setup
To run the code, follow these steps:
1. Install the required dependencies by executing `!pip install kaggle` in your Python environment.
2. Place your Kaggle API key (`kaggle.json`) in the same directory as the code.
3. Import the dataset using the Kaggle CLI by executing `!kaggle datasets download -d lakshmi25npathi/imdb-dataset-of-50k-movie-reviews`.
4. Extract the dataset files.
5. Install necessary Python libraries such as `numpy`, `pandas`, `nltk`, and `scikit-learn`.
6. Download NLTK's stopwords by executing `nltk.download('stopwords')`.

## Preprocessing
1. The dataset is loaded into a Pandas DataFrame.
2. Text preprocessing steps include:
   - Converting text to lowercase.
   - Removing non-alphabetic characters.
   - Tokenizing and stemming words using NLTK's Porter Stemmer.
   - Removing stopwords.
3. Preprocessed data is split into training and test sets.

## Feature Engineering
Text data is converted into numerical vectors using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization.

## Model Training
A logistic regression model is trained using the TF-IDF transformed training data.

## Model Evaluation
Model performance is evaluated using accuracy scores on both training and test datasets.

## Model Deployment
The trained model is saved to disk using pickle for future use.

## Future Predictions
To make predictions on new data:
1. Load the saved model using pickle.
2. Preprocess the new data.
3. Use the loaded model to predict sentiments.

## Files
- `README.md`: Documentation describing the project, dataset, setup, preprocessing, model training, evaluation, deployment, and future predictions.
- `IMDB Dataset.csv`: Original dataset containing movie reviews and sentiments.
- `test_data.csv`: CSV file containing preprocessed test data.
- `trained_model.sav`: Saved trained logistic regression model.

## Conclusion
This project demonstrates how to perform sentiment analysis on IMDB movie reviews using machine learning techniques. By training a logistic regression model on preprocessed text data, we can effectively classify movie reviews as positive or negative.

---
