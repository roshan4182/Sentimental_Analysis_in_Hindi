import pandas as pd
from indicnlp.tokenize import indic_tokenize

def tokenize_text(text):
    tokens = list(indic_tokenize.trivial_tokenize(str(text)))
    return ' '.join(tokens)

df = pd.read_csv('/Users/roshan1610/Desktop/sentiment-analysis-env/cleaned_data.csv')
df['cleaned_text'] = df['cleaned_text'].astype(str)
df['tokenized_text'] = df['cleaned_text'].apply(tokenize_text)
print(df.head())
df.to_csv('/Users/roshan1610/Desktop/sentiment-analysis-env/tokenized_data.csv', index=False)
