import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import joblib
import json
import numpy as np

class LangIDModel(nn.Module):
    def __init__(self, model_name, num_langs):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_langs)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(outputs.last_hidden_state[:, 0])
        logits = self.classifier(pooled)
        return logits

class StackedSentimentModel(nn.Module):
    def __init__(self, afrisenti_name, langid_model, num_langs, num_sentiment_classes):
        super().__init__()
        self.langid_model = langid_model
        self.sentiment_encoder = AutoModel.from_pretrained(afrisenti_name)
        hidden_size_sentiment = self.sentiment_encoder.config.hidden_size
        self.classifier = nn.Linear(hidden_size_sentiment + num_langs, num_sentiment_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, sentiment_input_ids, sentiment_attention_mask,
                langid_input_ids, langid_attention_mask, **kwargs):
        # Get language predictions
        langid_logits = self.langid_model(langid_input_ids, langid_attention_mask)
        langid_probs = torch.softmax(langid_logits, dim=1)
        
        # Get sentiment features
        sentiment_outputs = self.sentiment_encoder(input_ids=sentiment_input_ids, attention_mask=sentiment_attention_mask)
        sentiment_cls = self.dropout(sentiment_outputs.last_hidden_state[:, 0])
        
        # Combine features
        combined = torch.cat([sentiment_cls, langid_probs], dim=1)
        sentiment_logits = self.classifier(combined)
        
        return {"sentiment_logits": sentiment_logits, "langid_logits": langid_logits}

class EnsemblePredictor:
    def __init__(self, model_dir="ensemble_model"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_dir = model_dir
        
        # Load configuration
        with open(f"{model_dir}/config.json", "r") as f:
            self.config = json.load(f)
        
        # Load label mappings
        with open(f"{model_dir}/lang_labels.json", "r") as f:
            self.lang_labels = {int(k): v for k, v in json.load(f).items()}
        
        with open(f"{model_dir}/sentiment_labels.json", "r") as f:
            self.sentiment_labels = {int(k): v for k, v in json.load(f).items()}
        
        # Load tokenizers
        self.langid_tokenizer = AutoTokenizer.from_pretrained(f"{model_dir}/langid_tokenizer")
        self.sentiment_tokenizer = AutoTokenizer.from_pretrained(f"{model_dir}/sentiment_tokenizer")
        
        # Initialize and load models
        self.langid_model = LangIDModel(
            self.config["langid_model_name"], 
            self.config["num_langs"]
        )
        
        self.ensemble_model = StackedSentimentModel(
            self.config["afrisenti_model_name"],
            self.langid_model,
            self.config["num_langs"],
            self.config["num_sentiment_classes"]
        )
        
        # Load model weights
        self.langid_model.load_state_dict(torch.load(f"{model_dir}/langid_model.pt", map_location=self.device))
        self.ensemble_model.load_state_dict(torch.load(f"{model_dir}/ensemble_model.pt", map_location=self.device))
        
        # Move to device and set to eval mode
        self.langid_model.to(self.device)
        self.ensemble_model.to(self.device)
        self.langid_model.eval()
        self.ensemble_model.eval()
        
        print(f"Models loaded successfully on {self.device}")
        print(f"Available languages: {list(self.lang_labels.values())}")
        print(f"Available sentiments: {list(self.sentiment_labels.values())}")
    
    def predict(self, text, return_probabilities=False):
        """
        Predict language and sentiment for a given text
        
        Args:
            text (str): Input text to analyze
            return_probabilities (bool): Whether to return probabilities along with predictions
        
        Returns:
            dict: Predictions with language and sentiment
        """
        with torch.no_grad():
            # Tokenize text for both models
            langid_inputs = self.langid_tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                padding=True, 
                max_length=512
            ).to(self.device)
            
            sentiment_inputs = self.sentiment_tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                padding=True, 
                max_length=512
            ).to(self.device)
            
            # Get predictions
            outputs = self.ensemble_model(
                sentiment_input_ids=sentiment_inputs["input_ids"],
                sentiment_attention_mask=sentiment_inputs["attention_mask"],
                langid_input_ids=langid_inputs["input_ids"],
                langid_attention_mask=langid_inputs["attention_mask"]
            )
            
            # Process predictions
            sentiment_logits = outputs["sentiment_logits"]
            langid_logits = outputs["langid_logits"]
            
            sentiment_probs = torch.softmax(sentiment_logits, dim=1)
            langid_probs = torch.softmax(langid_logits, dim=1)
            
            # Get predicted classes
            sentiment_pred = torch.argmax(sentiment_probs, dim=1).item()
            langid_pred = torch.argmax(langid_probs, dim=1).item()
            
            # Convert to labels
            predicted_sentiment = self.sentiment_labels[sentiment_pred]
            predicted_language = self.lang_labels[langid_pred]
            
            result = {
                "text": text,
                "predicted_language": predicted_language,
                "predicted_sentiment": predicted_sentiment,
                "language_confidence": langid_probs[0][langid_pred].item(),
                "sentiment_confidence": sentiment_probs[0][sentiment_pred].item()
            }
            
            if return_probabilities:
                # Add all probabilities
                lang_probs_dict = {self.lang_labels[i]: prob.item() for i, prob in enumerate(langid_probs[0])}
                sent_probs_dict = {self.sentiment_labels[i]: prob.item() for i, prob in enumerate(sentiment_probs[0])}
                
                result["language_probabilities"] = lang_probs_dict
                result["sentiment_probabilities"] = sent_probs_dict
            
            return result
    
    def predict_batch(self, texts, return_probabilities=False):
        """
        Predict language and sentiment for multiple texts
        
        Args:
            texts (list): List of input texts
            return_probabilities (bool): Whether to return probabilities
        
        Returns:
            list: List of predictions
        """
        results = []
        for text in texts:
            result = self.predict(text, return_probabilities)
            results.append(result)
        return results

# Example usage
if __name__ == "__main__":
    # Initialize predictor
    predictor = EnsemblePredictor("ensemble_model")
    
    # Example texts (replace with your own)
    sample_texts = [
        "Hahaha hili zuri zaidi prosafe",
        "Àga tí ẹ̀ ń wò yìí ojúbọ Olúorógbó ní Ifẹ̀ ni wọ́n ti jí i, ó ńbẹ nílé ọ̀nà Britain; #BritishMuseum báyìí báyìí http://t.co/vFkAPejZYb",
        "ኦሮሚያ ለ ተወለዱ ወይም ረዥም ጊዜ ለ ኖሩ አማሮች ኦሮሚያ አገራቸው ነች:: እሱ ክርክር ውስጥ የሚገባ አይደለም::",
        "@user 🤣🤣🤣🤣🤣 bia choro mu ego ka mu je Enugu zutaragi okpa nnuka 😂😂😂",
        "በ3 መዝገብ የተከሰሱ 6 ግለሰቦች በፍርድ ቤት ውሳኔ እንዳገኙ ምክትል ጠቅላይ አቃቤ ህጉ ጠቅሰዋል፡፡  ሰበታ ላይ ቀላል ጉዳት በማድረስ ወንጀል ተከስሶ 6 ወር የተቀጣው ግለሰብ አንዱ ነው ብለዋል፡፡ 2/3",
        "عبروا عليهم لو كان اداوا واجبهم المهنى على احسن وجه الحكومة ما فتحاش خيرية"
    ]
    
    print("\n=== Single Predictions ===")
    for text in sample_texts:
        result = predictor.predict(text, return_probabilities=True)
        print(f"\nText: {result['text']}")
        print(f"Language: {result['predicted_language']} (confidence: {result['language_confidence']:.3f})")
        print(f"Sentiment: {result['predicted_sentiment']} (confidence: {result['sentiment_confidence']:.3f})")
    
    print("\n=== Batch Predictions ===")
    batch_results = predictor.predict_batch(sample_texts)
    for i, result in enumerate(batch_results):
        print(f"Text {i+1}: {result['predicted_language']} | {result['predicted_sentiment']}")