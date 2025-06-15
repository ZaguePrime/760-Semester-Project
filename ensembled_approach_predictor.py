import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import os

# === SETTINGS ===
model_base_path = "sentiment_model"
class_labels = ["negative", "neutral", "positive"]  # These must match your actual class names

# === INPUT ===
language = "sw"  # e.g., 'en', 'ha', 'am', 'sw', 'yo'
text = "Hii bidhaa ni bora sana!"  # your test input

# === Preprocess (prepend language token) ===
input_text = f"{language} [SEP] {text}"

# === Store results ===
results = {}

# === Loop through all class models ===
for label in class_labels:
    model_path = os.path.join(model_base_path, f"transformer_{label}", "model")
    tokenizer_path = os.path.join(model_base_path, f"transformer_{label}", "tokenizer")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True)

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)
        confidence = probs[0][1].item()  # Probability of class 1 (class == this sentiment)

    results[label] = confidence

# === Determine final prediction ===
predicted_class = max(results, key=results.get)
print(f"\n📝 Input Text: {text}")
print(f"🌍 Language: {language}")
print(f"\n🔮 Predicted Sentiment: {predicted_class.upper()}")
print("\n📊 Confidence Scores:")
for label, score in results.items():
    print(f" - {label}: {score:.4f}")
