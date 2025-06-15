import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding
)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import joblib
import os

# === Load TSV ===
df = pd.read_csv("all_languages_train_shuffled.tsv", sep="\t")

# === Ensure 'text', 'label', 'language' columns exist ===
df = df[["text", "label", "language"]].dropna()

# === Encode sentiment label column ===
sentiment_encoder = LabelEncoder()
df["label"] = sentiment_encoder.fit_transform(df["label"])

# === Save label encoder ===
os.makedirs("baseline_model", exist_ok=True)
joblib.dump(sentiment_encoder, "baseline_model/sentiment_encoder.pkl")

# === Train/test split ===
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

# === Convert to HuggingFace Datasets ===
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# === Tokenizer and Model ===
model_name = "Davlan/afro-xlmr-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize the text (without adding language info)
train_dataset = train_dataset.map(lambda x: tokenizer(x["text"], truncation=True), batched=True)
test_dataset = test_dataset.map(lambda x: tokenizer(x["text"], truncation=True), batched=True)

# Define model for classification
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(sentiment_encoder.classes_)
)

# === Define metrics ===
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# === Trainer arguments ===
training_args = TrainingArguments(
    output_dir="sentiment_model",
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    logging_dir="baseline_model/logs",
    load_best_model_at_end=True,
)

# === Trainer ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
)

# === Train and Save ===
trainer.train()
trainer.save_model("baseline_model/model")
tokenizer.save_pretrained("baseline_model/tokenizer")

# === Evaluate on test set ===
import torch
from sklearn.metrics import classification_report

# Get true labels
true_labels = test_dataset["label"]

# Predict
predictions = trainer.predict(test_dataset)
preds = torch.tensor(predictions.predictions).argmax(dim=-1).numpy()

# Decode labels
true_sentiments = sentiment_encoder.inverse_transform(true_labels)
predicted_sentiments = sentiment_encoder.inverse_transform(preds)

# === Print Report ===
print("\n=== Sentiment Analysis Results ===")
accuracy = accuracy_score(true_sentiments, predicted_sentiments)
print(f"Sentiment Analysis Accuracy: {accuracy:.4f}\n")

print("Sentiment Classification Report:")
print(classification_report(true_sentiments, predicted_sentiments, digits=4))
