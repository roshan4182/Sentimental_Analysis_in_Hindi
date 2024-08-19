import pandas as pd
import re


df = pd.read_csv('/Users/roshan1610/Downloads/output_combined_data_processed (1).csv')


print(df.head())


print(df.isnull().sum())


print(df['Label'].value_counts(normalize=True) * 100)




def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\w\s]', '', text)
    return text

df['cleaned_text'] = df['Sentence'].apply(clean_text)




