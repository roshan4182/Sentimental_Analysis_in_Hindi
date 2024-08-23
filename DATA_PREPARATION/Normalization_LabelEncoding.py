import pandas as pd

df = pd.read_csv('/Users/roshan1610/Desktop/sentiment-analysis-env/processed_data.csv')
df['normalized_text'] = df['processed_text'].str.lower()
print(df.head())  

label_mapping = {'neutral': 0, 'joy': 1, 'anger': 2, 'surprise': 3, 'sadness': 4, 'disgust': 5, 'fear': 6}
df['label'] = df['Label'].map(label_mapping)
df.to_csv('/Users/roshan1610/Desktop/sentiment-analysis-env/final_data.csv', index=False)
print(df.head())  
