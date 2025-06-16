import os
import pandas as pd
import joblib
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding
)
from datasets import Dataset
import numpy as np

# configuration
BASE_DIR = "../pipeline_cv"
SEPARATOR = " [SEP] "
os.makedirs(BASE_DIR, exist_ok=True)

# setup dataset
df = pd.read_csv("../language_datasets/all_languages_train_shuffled.tsv", sep="\t")
df = df[["text", "label", "language"]].dropna()

# setup cross validation
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
folds = list(kfold.split(df, df["label"]))

# encoders
lang_encoder = LabelEncoder()
lang_encoder.fit(df["language"])
sent_encoder = LabelEncoder()
sent_encoder.fit(df["label"])

# Model base
model_name = "Davlan/afro-xlmr-base"

# metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}

all_lang_preds, all_lang_true = [], []
all_sent_preds, all_sent_true = [], []

# folds
for fold, (train_idx, test_idx) in enumerate(folds):
    # setup fold directory
    print(f"\n==== Fold {fold+1}/5 ====")
    fold_dir = os.path.join(BASE_DIR, f"fold_{fold}")
    os.makedirs(fold_dir, exist_ok=True)

    # Language DataPrep and Model setup
    train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]

    lang_train_df = train_df[["text", "language"]].copy()
    lang_test_df = test_df[["text", "language"]].copy()
    lang_train_df["label"] = lang_encoder.transform(lang_train_df["language"])
    lang_test_df["label"] = lang_encoder.transform(lang_test_df["language"])

    lang_tokenizer = AutoTokenizer.from_pretrained(model_name)
    lang_train_ds = Dataset.from_pandas(lang_train_df)
    lang_test_ds = Dataset.from_pandas(lang_test_df)
    lang_train_ds = lang_train_ds.map(lambda x: lang_tokenizer(x["text"], truncation=True), batched=True)
    lang_test_ds = lang_test_ds.map(lambda x: lang_tokenizer(x["text"], truncation=True), batched=True)

    lang_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(lang_encoder.classes_))
    lang_args = TrainingArguments(
        output_dir=os.path.join(fold_dir, "lang_model"),
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=32,
        num_train_epochs=1,
        logging_dir=os.path.join(fold_dir, "lang_logs"),
        load_best_model_at_end=True
    )
    lang_trainer = Trainer(
        model=lang_model,
        args=lang_args,
        train_dataset=lang_train_ds,
        eval_dataset=lang_test_ds,
        tokenizer=lang_tokenizer,
        data_collator=DataCollatorWithPadding(lang_tokenizer),
        compute_metrics=compute_metrics
    )
    # train language model
    lang_trainer.train()

    # Sentiment DataPrep and Model setup
    sent_train_df = train_df.copy()
    sent_test_df = test_df.copy()
    sent_train_df["label"] = sent_encoder.transform(sent_train_df["label"])
    sent_test_df["label"] = sent_encoder.transform(sent_test_df["label"])
    sent_train_df["text"] = sent_train_df["language"] + SEPARATOR + sent_train_df["text"]
    sent_test_df["text"] = sent_test_df["language"] + SEPARATOR + sent_test_df["text"]

    sent_tokenizer = AutoTokenizer.from_pretrained(model_name)
    sent_train_ds = Dataset.from_pandas(sent_train_df)
    sent_test_ds = Dataset.from_pandas(sent_test_df)
    sent_train_ds = sent_train_ds.map(lambda x: sent_tokenizer(x["text"], truncation=True), batched=True)
    sent_test_ds = sent_test_ds.map(lambda x: sent_tokenizer(x["text"], truncation=True), batched=True)

    sent_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(sent_encoder.classes_))
    sent_args = TrainingArguments(
        output_dir=os.path.join(fold_dir, "sent_model"),
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=32,
        num_train_epochs=3,
        logging_dir=os.path.join(fold_dir, "sent_logs"),
        load_best_model_at_end=True
    )
    sent_trainer = Trainer(
        model=sent_model,
        args=sent_args,
        train_dataset=sent_train_ds,
        eval_dataset=sent_test_ds,
        tokenizer=sent_tokenizer,
        data_collator=DataCollatorWithPadding(sent_tokenizer),
        compute_metrics=compute_metrics
    )
    # train sentiment model
    sent_trainer.train()

    # evaluate
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lang_model.to(device)
    sent_model.to(device)

    lang_preds, sent_preds, lang_true, sent_true = [], [], [], []

    for i in range(len(test_df)):
        raw_text = test_df.iloc[i]["text"]
        true_lang = test_df.iloc[i]["language"]
        true_sent = test_df.iloc[i]["label"]

        lang_input = lang_tokenizer(raw_text, return_tensors="pt", truncation=True).to(device)
        with torch.no_grad():
            lang_logits = lang_model(**lang_input).logits
        pred_lang_id = lang_logits.argmax(dim=-1).item()
        pred_lang = lang_encoder.inverse_transform([pred_lang_id])[0]

        new_text = f"{pred_lang}{SEPARATOR}{raw_text}"
        sent_input = sent_tokenizer(new_text, return_tensors="pt", truncation=True).to(device)
        with torch.no_grad():
            sent_logits = sent_model(**sent_input).logits
        pred_sent_id = sent_logits.argmax(dim=-1).item()

        lang_preds.append(pred_lang_id)
        lang_true.append(lang_encoder.transform([true_lang])[0])
        sent_preds.append(pred_sent_id)
        sent_true.append(sent_encoder.transform([true_sent])[0])

    all_lang_preds.extend(lang_preds)
    all_lang_true.extend(lang_true)
    all_sent_preds.extend(sent_preds)
    all_sent_true.extend(sent_true)

# pretty print results
print("\n=== Final Cross-Validation Results ===")
print("\n-- Language Identification --")
print(f"Accuracy: {accuracy_score(all_lang_true, all_lang_preds):.4f}")
print(classification_report(all_lang_true, all_lang_preds, target_names=lang_encoder.classes_, zero_division=0))

print("\n-- Sentiment Analysis --")
print(f"Accuracy: {accuracy_score(all_sent_true, all_sent_preds):.4f}")
print(classification_report(all_sent_true, all_sent_preds, target_names=sent_encoder.classes_, zero_division=0))