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

# directory configuration
DIR = "../baseline_model_cv"
os.makedirs(DIR, exist_ok=True)

# preproccess data
df = pd.read_csv("../language_datasets/all_languages_train_shuffled.tsv", sep="\t")
df = df[["text", "label", "language"]].dropna()

sentiment_encoder = LabelEncoder()
df["label"] = sentiment_encoder.fit_transform(df["label"])
num_labels = len(sentiment_encoder.classes_)

joblib.dump(sentiment_encoder, os.path.join(DIR, "sentiment_encoder.pkl"))

# model setup with cross validation
model_name = "Davlan/afro-xlmr-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

all_preds = []
all_trues = []
fold_metrics = []

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(skf.split(df["text"], df["label"])):
    print(f"\n=== Fold {fold + 1} ===")

    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]

    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)

    train_ds = train_ds.map(lambda x: tokenizer(x["text"], truncation=True), batched=True)
    val_ds = val_ds.map(lambda x: tokenizer(x["text"], truncation=True), batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

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

    # Train and evaluate
    trainer.train()

    predictions = trainer.predict(val_ds)
    preds = torch.tensor(predictions.predictions).argmax(dim=-1).numpy()
    true_labels = np.array(val_ds["label"])

    all_preds.extend(preds)
    all_trues.extend(true_labels)

    acc = accuracy_score(true_labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(true_labels, preds, average="weighted")

    # pretty print metrics for fold
    print(f"Fold {fold+1} Metrics:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")

    fold_metrics.append({"accuracy": acc, "precision": prec, "recall": rec, "f1": f1})

# final evaluation across all folds
print("\n=== Final Cross-Validation Results ===")

avg_metrics = {
    metric: np.mean([fm[metric] for fm in fold_metrics])
    for metric in ["accuracy", "precision", "recall", "f1"]
}

for k, v in avg_metrics.items():
    print(f"Avg {k.capitalize()}: {v:.4f}")

true_sentiments = sentiment_encoder.inverse_transform(all_trues)
predicted_sentiments = sentiment_encoder.inverse_transform(all_preds)

print("\nFull Classification Report:")
print(classification_report(true_sentiments, predicted_sentiments, digits=4))
