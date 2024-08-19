import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('/Users/roshan1610/Desktop/sentiment-analysis-env/final_data.csv')

# Assuming the label column is named 'label'
label_counts = df['label'].value_counts()

# Plotting the class distribution
plt.figure(figsize=(10, 6))
label_counts.plot(kind='bar', color='skyblue')
plt.title('Class Distribution', fontweight='bold')
plt.xlabel('Class', fontweight='bold')
plt.ylabel('Number of Samples', fontweight='bold')
plt.xticks(rotation=0, fontweight='bold')
plt.yticks(fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
