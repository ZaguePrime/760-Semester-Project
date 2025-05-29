import torch
import joblib
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn

class MultiTaskModel(nn.Module):
    def __init__(self, base_model_name, num_langs, num_sentiments):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name)
        hidden_size = self.encoder.config.hidden_size
        self.lang_classifier = nn.Linear(hidden_size, num_langs)
        self.sentiment_classifier = nn.Linear(hidden_size, num_sentiments)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]
        lang_logits = self.lang_classifier(pooled)
        sentiment_logits = self.sentiment_classifier(pooled)
        return lang_logits, sentiment_logits

# Load tokenizer and encoders
model_dir = "multitask_afrosenti_model"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
lang_encoder = joblib.load("lang_encoder.pkl")
sentiment_encoder = joblib.load("sentiment_encoder.pkl")

# Load model
num_langs = len(lang_encoder.classes_)
num_sentiments = len(sentiment_encoder.classes_)
model = MultiTaskModel(model_dir, num_langs, num_sentiments)
model.load_state_dict(torch.load(f"{model_dir}/pytorch_model.bin", map_location="cpu"))
model.eval()

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        lang_logits, sentiment_logits = model(**inputs)
        lang_pred = torch.argmax(lang_logits, dim=1).item()
        sentiment_pred = torch.argmax(sentiment_logits, dim=1).item()

    lang_label = lang_encoder.inverse_transform([lang_pred])[0]
    sentiment_label = sentiment_encoder.inverse_transform([sentiment_pred])[0]
    return lang_label, sentiment_label

# Example usage
if __name__ == "__main__":
    tweet = ""
    lang, sentiment = predict(tweet)
    print(f"Language: {lang} | Sentiment: {sentiment}")
