## Sentiment Analysis in Hindi Language

## Overview

This project explores sentiment analysis in the Hindi language, aiming to classify text into various sentiment categories. A significant challenge encountered was the scarcity of publicly available datasets for Hindi sentiment analysis. Despite these limitations, the project demonstrates the viability of employing machine learning models for sentiment classification in Hindi, although data constraints have affected accuracy.

## Project Structure

Data Preparation: Contains scripts and procedures for cleaning, normalizing, tokenizing, and preparing text data for modeling.
Model Training: Scripts for training different machine learning models used in sentiment analysis.
Results: A directory housing evaluation metrics and performance results of the models.
Graph Scripts: Scripts for generating visualizations during the analysis.
Pretrained Model: A directory containing the pre-trained model weights.
Models Used

## The following machine learning models were trained and evaluated on the dataset:

1. BERT (Bidirectional Encoder Representations from Transformers)
Accuracy: 49.0%
F1-Score (weighted): 0.40
Results: BERT, chosen for its state-of-the-art performance in NLP tasks, delivered limited accuracy due to the small dataset size.

2. Logistic Regression
Accuracy: 49.0%
F1-Score (weighted): 0.40
Remarks: Logistic Regression served as a baseline model, providing competitive results despite its simplicity.

3. Support Vector Machine (SVM)
Accuracy: 51.0%
F1-Score (weighted): 0.39
Remarks: SVM slightly outperformed Logistic Regression but struggled with the limited data.

4. XGBoost
Accuracy: 49.0%
F1-Score (weighted): 0.41
Remarks: XGBoost, known for its strength in structured data, faced challenges due to the textual nature of the data and its limited size.

## Challenges and Limitations

1. Data Scarcity
The primary challenge was the lack of available datasets for Hindi sentiment analysis. The small dataset size adversely affected the models' ability to generalize, leading to lower accuracy and F1-scores. Nevertheless, this project highlights the potential for sentiment analysis in underrepresented languages like Hindi.

2. Model Performance
Due to data constraints, all models achieved accuracy below 55%. BERT, despite its advanced capabilities, could not reach its full potential because of insufficient training data. Traditional models like Logistic Regression and SVM also exhibited lower performance.

## Conclusion

This project offers an initial exploration into sentiment analysis in Hindi, providing insights into the challenges and opportunities of applying machine learning models to underrepresented languages. The relatively low accuracy reflects the current limitations in data availability rather than the models' capabilities. With access to more extensive and diverse datasets, these models could achieve significantly higher accuracy and better generalization.

## Future Work

To build on this project, future efforts should focus on:

1. Data Augmentation: Expanding the dataset through various data augmentation techniques or collecting additional labeled data.
2. Advanced Models: Exploring more sophisticated models or ensemble techniques to improve accuracy.
3. Transfer Learning: Leveraging pre-trained models on large multilingual datasets and fine-tuning them on Hindi text.

## Repository Contents

DATA_PREPARATION: Scripts for data cleaning, tokenization, and normalization.
MODEL_TRAINING: Training scripts for various models.
RESULTS: Output files and performance metrics of the trained models.
GRAPH_SCRIPTS: Code for generating visualizations.
sentiment_model: Directory containing pre-trained model weights.
TOKENIZATION: Tokenization scripts for text preprocessing.
