import os
import torch
import pandas as pd
import numpy as np
import joblib

from datasets import Dataset
from sklearn.model_selection import StratifiedKFold
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

# === Configurable base directory for saving model stuff ===
DIR = "../baseline_model_cv"
os.makedirs(DIR, exist_ok=True)

# === Load and prepare dataset ===
df = pd.read_csv("../language_datasets/all_languages_train_shuffled.tsv", sep="\t")
df = df[["text", "label", "language"]].dropna()

# === Encode sentiment labels ===
sentiment_encoder = LabelEncoder()
df["label"] = sentiment_encoder.fit_transform(df["label"])
num_labels = len(sentiment_encoder.classes_)

# Save encoder
joblib.dump(sentiment_encoder, os.path.join(DIR, "sentiment_encoder.pkl"))

# === Tokenizer and model ===
model_name = "Davlan/afro-xlmr-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# === Metric storage ===
all_preds = []
all_trues = []
fold_metrics = []

# === Cross-validation setup ===
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(df["text"], df["label"])):
    print(f"\n=== Fold {fold + 1} ===")

    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]

    # Convert to HuggingFace Datasets
    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)

    # Tokenize
    train_ds = train_ds.map(lambda x: tokenizer(x["text"], truncation=True), batched=True)
    val_ds = val_ds.map(lambda x: tokenizer(x["text"], truncation=True), batched=True)

    # Define model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    # Training arguments
    fold_output_dir = os.path.join(DIR, f"fold_{fold + 1}")
    os.makedirs(fold_output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=fold_output_dir,
        eval_strategy="epoch",
        save_strategy="no",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        logging_dir=os.path.join(fold_output_dir, "logs"),
        load_best_model_at_end=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
    )

    trainer.train()

    # Predict
    predictions = trainer.predict(val_ds)
    preds = torch.tensor(predictions.predictions).argmax(dim=-1).numpy()
    true_labels = np.array(val_ds["label"])

    # Store all predictions
    all_preds.extend(preds)
    all_trues.extend(true_labels)

    # Compute metrics
    acc = accuracy_score(true_labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(true_labels, preds, average="weighted")

    print(f"Fold {fold+1} Metrics:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")

    fold_metrics.append({"accuracy": acc, "precision": prec, "recall": rec, "f1": f1})

# === Aggregate evaluation ===
print("\n=== Final Cross-Validation Results ===")

# Average metrics
avg_metrics = {
    metric: np.mean([fm[metric] for fm in fold_metrics])
    for metric in ["accuracy", "precision", "recall", "f1"]
}

for k, v in avg_metrics.items():
    print(f"Avg {k.capitalize()}: {v:.4f}")

# Decode final predictions for full classification report
true_sentiments = sentiment_encoder.inverse_transform(all_trues)
predicted_sentiments = sentiment_encoder.inverse_transform(all_preds)

print("\nFull Classification Report:")
print(classification_report(true_sentiments, predicted_sentiments, digits=4))
