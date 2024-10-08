import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Loading  dataset
df = pd.read_csv('/Users/roshan1610/Desktop/sentiment-analysis-env/final_data.csv')

# Ensuring all text data is in string format
df['normalized_text'] = df['normalized_text'].astype(str)

# Encoding the labels as integers
df['label'] = df['label'].astype(int)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['normalized_text'], df['label'], test_size=0.2, random_state=42)

# Vectorizing the text data using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Training Logistic Regression Model
lr_model = LogisticRegression()
lr_model.fit(X_train_tfidf, y_train)

# Making predictions
y_pred_lr = lr_model.predict(X_test_tfidf)

# Printing classification report
print("Logistic Regression Classification Report")
print(classification_report(y_test, y_pred_lr))
# Saving Logistic Regression results
with open('/Users/roshan1610/Desktop/sentiment-analysis-env/logistic_regression_results.txt', 'w') as f:
    f.write("Logistic Regression Classification Report\n")
    f.write(classification_report(y_test, y_pred_lr))
