import pandas as pd
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf
from sklearn.model_selection import KFold
import numpy as np

df = pd.read_csv('/Users/roshan1610/Desktop/sentiment-analysis-env/final_data.csv')
df['normalized_text'] = df['normalized_text'].astype(str)
df['label'] = df['label'].astype(int)
model_name = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
encoded_inputs = tokenizer(df['normalized_text'].tolist(), padding=True, truncation=True, max_length=128, return_tensors="tf")

input_ids = encoded_inputs['input_ids'].numpy()
kf = KFold(n_splits=5)
accuracies = []

for train_index, test_index in kf.split(input_ids):
    X_train, X_test = input_ids[train_index], input_ids[test_index]
    y_train, y_test = df['label'].values[train_index], df['label'].values[test_index]

    model = TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=7)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5), 
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
                  metrics=['accuracy'])

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X_train)).batch(16)
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(16)

    model.fit(train_dataset, epochs=3, verbose=1)
    loss, accuracy = model.evaluate(test_dataset)
    accuracies.append(accuracy)

print(f"Average Accuracy: {np.mean(accuracies) * 100:.2f}%")
