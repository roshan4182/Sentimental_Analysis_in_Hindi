import pandas as pd
stopwords = set(['है', 'में', 'के', 'यह', 'और', 'को'])
df = pd.read_csv('/Users/roshan1610/Desktop/sentiment-analysis-env/tokenized_data.csv')
df['tokenized_text'] = df['tokenized_text'].fillna('')
def remove_stopwords(text):
    return ' '.join([word for word in text.split() if word not in stopwords])
df['processed_text'] = df['tokenized_text'].apply(remove_stopwords)
df.to_csv('/Users/roshan1610/Desktop/sentiment-analysis-env/processed_data.csv', index=False)
print(df.head())  
