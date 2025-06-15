from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib
import torch

class LangSentimentPipeline:
    def __init__(self):
        # Load everything
        self.lang_tokenizer = AutoTokenizer.from_pretrained("langid_model/tokenizer")
        self.lang_model = AutoModelForSequenceClassification.from_pretrained("langid_model/model")
        self.lang_encoder = joblib.load("langid_model/lang_encoder.pkl")
        
        self.sent_tokenizer = AutoTokenizer.from_pretrained("sentiment_model/tokenizer")
        self.sent_model = AutoModelForSequenceClassification.from_pretrained("sentiment_model/model")
        self.sent_encoder = joblib.load("sentiment_model/sentiment_encoder.pkl")

    def __call__(self, text):
        lang = self.predict_language(text)
        sentiment = self.predict_sentiment(text, lang)
        return {"language": lang, "sentiment": sentiment}

    def predict_language(self, text):
        inputs = self.lang_tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = self.lang_model(**inputs)
        pred_id = outputs.logits.argmax(dim=-1).item()
        return self.lang_encoder.inverse_transform([pred_id])[0]

    def predict_sentiment(self, text, lang):
        combined_input = f"{lang} [SEP] {text}"
        inputs = self.sent_tokenizer(combined_input, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = self.sent_model(**inputs)
        pred_id = outputs.logits.argmax(dim=-1).item()
        return self.sent_encoder.inverse_transform([pred_id])[0]

# Usage
pipeline = LangSentimentPipeline()
print(pipeline("Tani jeun ti aja un juru, Ebenezer Babatope na un soro.... ara awon ojelu ti won tan @user de iparun re o https://t.co/BVAPVTwIXR"))  # -> {'language': 'de', 'sentiment': 'positive'}
