import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('/Users/roshan1610/Desktop/sentiment-analysis-env/final_data.csv')

# Drop rows where normalized_text is NaN or None
df = df.dropna(subset=['normalized_text'])

# Combine all text data into a single string
text_data = ' '.join(df['normalized_text'].astype(str).tolist())

# Path to a font that supports Devanagari script
font_path = '/Users/roshan1610/Desktop/sentiment-analysis-env/Noto_Sans_Devanagari/NotoSansDevanagari-VariableFont_wdth,wght.ttf'  # Replace with the actual path to your Devanagari font

# Generate the word cloud
wordcloud = WordCloud(
    font_path=font_path, 
    width=800, 
    height=400, 
    background_color='white', 
    max_words=100,  # You can adjust this to focus on the most frequent words
    colormap='Blues', 
    contour_color='steelblue', 
    contour_width=1
).generate(text_data)

# Plot the word cloud
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('Word Cloud of Text Data', fontweight='bold')
plt.axis('off')  # Hide the axes
plt.show()
