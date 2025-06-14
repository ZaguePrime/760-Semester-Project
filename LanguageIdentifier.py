import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline

# --- Load Data from TSV ---
tsv_file = 'all_languages_train_shuffled.tsv'  # Replace with your actual file
df = pd.read_csv(tsv_file, sep='\t')

# Only keep the necessary columns
df = df[['text', 'language']].dropna()

# Shuffle the dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# --- Train/Test Split ---
X = df['text']
y = df['language']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# --- Pipeline ---
model_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
    ('classifier', MultinomialNB())
])

# --- Train Model ---
print("Training the model...")
model_pipeline.fit(X_train, y_train)
print("Model training complete.")

# --- Evaluate ---
y_pred = model_pipeline.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# --- Save Model ---
model_filename = 'final/language_classification_model.joblib'
joblib.dump(model_pipeline, model_filename)
print(f"\nModel saved as '{model_filename}'")
