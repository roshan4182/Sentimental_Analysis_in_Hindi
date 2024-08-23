import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from sklearn.model_selection import train_test_split

df = pd.read_csv('/Users/roshan1610/Desktop/sentiment-analysis-env/final_data.csv')
df['normalized_text'] = df['normalized_text'].astype(str)
df['label'] = df['label'].astype(int)

model_name = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

encoded_inputs = tokenizer(df['normalized_text'].tolist(), padding=True, truncation=True, max_length=128, return_tensors="tf")
input_ids = encoded_inputs['input_ids'].numpy()

X_train, X_test, y_train, y_test = train_test_split(input_ids, df['label'], test_size=0.2, random_state=42)
model = TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=7)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5), loss=loss_fn, metrics=['accuracy'])

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X_train)).batch(16)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(16)

history = model.fit(train_dataset, epochs=3, validation_data=test_dataset)

# Plotting the training and validation accuracy
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy', fontweight='bold')
plt.xlabel('Epoch', fontweight='bold')
plt.ylabel('Accuracy', fontweight='bold')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# Plotting the training and validation loss
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss', fontweight='bold')
plt.xlabel('Epoch', fontweight='bold')
plt.ylabel('Loss', fontweight='bold')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()
