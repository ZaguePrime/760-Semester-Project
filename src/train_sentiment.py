import os
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

# === Configurable model directory ===
DIR = "../sentiment_model"
os.makedirs(DIR, exist_ok=True)

# === Load TSV ===
df = pd.read_csv("../language_datasets/all_languages_train_shuffled.tsv", sep="\t")

# === Ensure 'text', 'label', 'language' columns exist ===
df = df[["text", "label", "language"]].dropna()

# === Encode sentiment label column ===
sentiment_encoder = LabelEncoder()
df["label"] = sentiment_encoder.fit_transform(df["label"])

# === Include language as a feature by prepending to text ===
df["text"] = df["language"] + " [SEP] " + df["text"]

# === Save label encoder ===
joblib.dump(sentiment_encoder, os.path.join(DIR, "sentiment_encoder.pkl"))

# === Train/test split ===
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

# === Convert to HuggingFace Datasets ===
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# === Tokenizer and Model ===
model_name = "Davlan/afro-xlmr-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize the text (with truncation)
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
    output_dir=DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    logging_dir=os.path.join(DIR, "logs"),
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
trainer.save_model(os.path.join(DIR, "model"))
tokenizer.save_pretrained(os.path.join(DIR, "tokenizer"))
