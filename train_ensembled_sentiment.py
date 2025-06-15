import pandas as pd
import os
import joblib
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# === Load data ===
df = pd.read_csv("all_languages_train_shuffled.tsv", sep="\t")
df = df[["text", "label", "language"]].dropna()

# === Get unique sentiment classes ===
classes = df["label"].unique()
os.makedirs("sentiment_model", exist_ok=True)

# === Loop through each class (One-vs-All) ===
for target_label in classes:
    print(f"\n🔁 Training model for sentiment class: {target_label}")

    # Binary relabeling: 1 if this class, 0 otherwise
    df_bin = df.copy()
    df_bin["binary_label"] = (df_bin["label"] == target_label).astype(int)

    # Prepend language to text
    df_bin["text"] = df_bin["language"] + " [SEP] " + df_bin["text"]

    # Stratified split
    train_df, test_df = train_test_split(
        df_bin,
        test_size=0.2,
        stratify=df_bin["binary_label"],
        random_state=42
    )

    # HuggingFace Dataset
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    # Tokenizer
    model_name = "Davlan/afro-xlmr-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset = train_dataset.map(lambda x: tokenizer(x["text"], truncation=True), batched=True)
    test_dataset = test_dataset.map(lambda x: tokenizer(x["text"], truncation=True), batched=True)

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Metrics
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

    # Training args
    label_name = str(target_label).lower()
    output_dir = f"sentiment_model/transformer_{label_name}"
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        logging_dir=os.path.join(output_dir, "logs"),
        load_best_model_at_end=True,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    # Train and save
    trainer.train()
    trainer.save_model(os.path.join(output_dir, "model"))
    tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))
    print(f"Model saved for class '{target_label}'")

print("\nAll one-vs-all sentiment transformers trained.")
