import pandas as pd
import re
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split

# Load dataset
file_path = "train.csv"  # Change this if needed
df = pd.read_csv(file_path)

# Preprocessing function
def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower().strip()  # Convert to lowercase
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

# Apply preprocessing
df["text"] = df["text"].apply(preprocess_text)
df["gloss"] = df["gloss"].apply(preprocess_text)

# 🛠️ Split dataset into training (80%) and validation (20%)
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Convert to Hugging Face Dataset format
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# 🛠️ Tokenization
tokenizer = T5Tokenizer.from_pretrained("t5-small")

def tokenize_function(examples):
    inputs = tokenizer(examples["text"], max_length=128, padding="max_length", truncation=True)
    targets = tokenizer(examples["gloss"], max_length=128, padding="max_length", truncation=True)
    inputs["labels"] = targets["input_ids"]
    return inputs

# Tokenize datasets
train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# Load Pretrained T5 Model
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# 🛠️ Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",  # Evaluate every epoch
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    save_steps=1000,
    save_total_limit=1,
    logging_dir="./logs",
    logging_steps=200,
)

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,  # ✅ Validation dataset added
)

# 🚀 Start training
print("🚀 Training started...")
trainer.train()
print("✅ Training completed!")

# Save the trained model & tokenizer
model.save_pretrained("saved_model")
tokenizer.save_pretrained("saved_model")
print("✅ Model saved successfully in 'saved_model' folder.")
