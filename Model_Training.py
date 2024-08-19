import pandas as pd
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# Load the final preprocessed data
df = pd.read_csv('/Users/roshan1610/Desktop/sentiment-analysis-env/final_data.csv')

# Load pre-trained mBERT tokenizer
model_name = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Encode the text
encoded_inputs = tokenizer(df['normalized_text'].tolist(), padding=True, truncation=True, max_length=128, return_tensors="tf")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(encoded_inputs['input_ids'], df['label'], test_size=0.2, random_state=42)

# Load pre-trained mBERT model for sequence classification
model = TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=7)

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X_train)).batch(16)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(16)

# Compile and train the model
model.compile(optimizer=Adam(learning_rate=3e-5), loss=model.compute_loss, metrics=['accuracy'])
history = model.fit(train_dataset, epochs=3, validation_data=test_dataset)

# Save the trained model
model.save_pretrained('/Users/roshan1610/Desktop/sentiment-analysis-env/sentiment_model')

# Evaluate the model
loss, accuracy = model.evaluate(test_dataset)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
