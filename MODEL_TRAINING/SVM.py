import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Loading  dataset
df = pd.read_csv('/Users/roshan1610/Desktop/sentiment-analysis-env/final_data.csv')

# Dropping rows with missing 'normalized_text'
df = df.dropna(subset=['normalized_text'])

# Splitting into features and labels
X = df['normalized_text']
y = df['label']

# Splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Filling NaN values with an empty string
X_train = X_train.fillna('')
X_test = X_test.fillna('')

# Converting text data into TF-IDF features
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Supportting Vector Machine (SVM) Model
svm_model = SVC()
svm_model.fit(X_train_tfidf, y_train)
y_pred_svm = svm_model.predict(X_test_tfidf)
print("SVM Classification Report")
print(classification_report(y_test, y_pred_svm))
# at last saving  SVM results
with open('/Users/roshan1610/Desktop/sentiment-analysis-env/svm_results.txt', 'w') as f:
    f.write("SVM Classification Report\n")
    f.write(classification_report(y_test, y_pred_svm))
