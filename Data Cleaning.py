import re
import pandas as pd

df = pd.read_csv('/Users/roshan1610/Downloads/output_combined_data_processed (1).csv')



def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\w\s]', '', text)
    return text

df['cleaned_text'] = df['Sentence'].apply(clean_text)
print(df.head())  # Check if the 'cleaned_text' column is present

df.to_csv('/Users/roshan1610/Desktop/sentiment-analysis-env/cleaned_data.csv', index=False)
print(df['cleaned_text'].isna().sum())

