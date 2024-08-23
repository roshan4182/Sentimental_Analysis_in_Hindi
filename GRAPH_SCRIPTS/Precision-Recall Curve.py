import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv('/Users/roshan1610/Desktop/sentiment-analysis-env/final_data.csv')
df['normalized_text'] = df['normalized_text'].astype(str)
df['label'] = df['label'].astype(int)

model_name = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
encoded_inputs = tokenizer(df['normalized_text'].tolist(), padding=True, truncation=True, max_length=128, return_tensors="tf")
input_ids = encoded_inputs['input_ids'].numpy()
X_train, X_test, y_train, y_test = train_test_split(input_ids, df['label'], test_size=0.2, random_state=42)

model_path = '/Users/roshan1610/Desktop/sentiment-analysis-env/sentiment_model'
print(f"Loading model from: {model_path}")
model = TFAutoModelForSequenceClassification.from_pretrained(model_path)

y_probs = model.predict(X_test).logits
y_probs = tf.nn.softmax(y_probs, axis=1).numpy()
class_id = 1  
y_test_class = np.where(y_test == class_id, 1, 0) 
y_probs_class = y_probs[:, class_id]  


precision, recall, thresholds = precision_recall_curve(y_test_class, y_probs_class)
average_precision = average_precision_score(y_test_class, y_probs_class)

# Plotting the Precision-Recall curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='.', label=f'Class {class_id} (AP = {average_precision:.2f})')
plt.xlabel('Recall', fontweight='bold')
plt.ylabel('Precision', fontweight='bold')
plt.title('Precision-Recall Curve', fontweight='bold')
plt.legend(loc='lower left')
plt.grid(True)
plt.show()
