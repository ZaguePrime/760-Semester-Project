import os
import torch
import pandas as pd
import numpy as np
import joblib

from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_recall_fscore_support
)

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding
)

# === Directory configuration ===
DIR = "../baseline_model"

# === Load and prepare dataset ===
df = pd.read_csv("../language_datasets/all_languages_train_shuffled.tsv", sep="\t")
df = df[["text", "label", "language"]].dropna()

# === Encode sentiment labels ===
sentiment_encoder = LabelEncoder()
df["label"] = sentiment_encoder.fit_transform(df["label"])
num_labels = len(sentiment_encoder.classes_)

# Save encoder
os.makedirs(DIR, exist_ok=True)
joblib.dump(sentiment_encoder, os.path.join(DIR, "sentiment_encoder.pkl"))

# === Tokenizer and model ===
model_name = "Davlan/afro-xlmr-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# === Train-test split (80/20) ===
train_df, test_df = train_test_split(
    df, 
    test_size=0.2, 
    random_state=42, 
    stratify=df["label"]
)

print(f"Training set size: {len(train_df)}")
print(f"Test set size: {len(test_df)}")

# Convert to HuggingFace Datasets
train_ds = Dataset.from_pandas(train_df)
test_ds = Dataset.from_pandas(test_df)

# Tokenize
train_ds = train_ds.map(lambda x: tokenizer(x["text"], truncation=True), batched=True)
test_ds = test_ds.map(lambda x: tokenizer(x["text"], truncation=True), batched=True)

# Define model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

# Training arguments
training_args = TrainingArguments(
    output_dir=DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    logging_dir=os.path.join(DIR, "logs"),
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
)

# Train the model
print("\n=== Training Model ===")
trainer.train()

# Save the final model and tokenizer
print(f"\n=== Saving Model to {DIR} ===")
trainer.save_model(DIR)
tokenizer.save_pretrained(DIR)

# === Final Evaluation ===
print("\n=== Final Evaluation ===")
predictions = trainer.predict(test_ds)
preds = torch.tensor(predictions.predictions).argmax(dim=-1).numpy()
true_labels = np.array(test_ds["label"])

# Compute metrics
acc = accuracy_score(true_labels, preds)
prec, rec, f1, _ = precision_recall_fscore_support(true_labels, preds, average="weighted")

print(f"Test Set Metrics:")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-score:  {f1:.4f}")

# Decode predictions for full classification report
true_sentiments = sentiment_encoder.inverse_transform(true_labels)
predicted_sentiments = sentiment_encoder.inverse_transform(preds)

print("\nFull Classification Report:")
print(classification_report(true_sentiments, predicted_sentiments, digits=4))

# Save test results
results = {
    "accuracy": acc,
    "precision": prec,
    "recall": rec,
    "f1": f1,
    "true_labels": true_labels.tolist(),
    "predictions": preds.tolist(),
    "true_sentiments": true_sentiments.tolist(),
    "predicted_sentiments": predicted_sentiments.tolist()
}

joblib.dump(results, os.path.join(DIR, "test_results.pkl"))
print(f"\nResults saved to {os.path.join(DIR, 'test_results.pkl')}")