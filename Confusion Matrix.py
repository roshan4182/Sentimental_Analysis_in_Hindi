import pandas as pd
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


df = pd.read_csv('/Users/roshan1610/Desktop/sentiment-analysis-env/final_data.csv')

df['normalized_text'] = df['normalized_text'].astype(str)


df['label'] = df['label'].astype(int)


model_name = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)


encoded_inputs = tokenizer(df['normalized_text'].tolist(), padding=True, truncation=True, max_length=128, return_tensors="tf")


input_ids = encoded_inputs['input_ids'].numpy()


X_train, X_test, y_train, y_test = train_test_split(input_ids, df['label'], test_size=0.2, random_state=42)


model = TFAutoModelForSequenceClassification.from_pretrained('/Users/roshan1610/Desktop/sentiment-analysis-env/sentiment_model')
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5), loss=loss_fn, metrics=['accuracy'])


y_pred = model.predict(X_test).logits
y_pred = tf.argmax(y_pred, axis=1).numpy()


cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix', fontweight='bold')
plt.show()


loss, accuracy = model.evaluate(tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(16))
print(f"\033[1mTest Accuracy: {accuracy * 100:.2f}%\033[0m")
