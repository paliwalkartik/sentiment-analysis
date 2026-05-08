import pandas as pd
import numpy as np
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text

print("Loading data...")
# Read data
df = pd.read_csv("data.csv")

print("Cleaning text...")
df['cleaned'] = df['Text'].apply(clean_text)

# Convert Text -> Numbers (Vectorization)
print("Vectorizing text...")
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned']).toarray()
y = df['Sentiment']

# Train-Test Split (handle edge case for very small sample data)
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
except ValueError:
    # If dataset is too small, just train and test on the same data for demonstration
    X_train, X_test, y_train, y_test = X, X, y, y

# Model Training
print("Training model...")
model = LogisticRegression()
model.fit(X_train, y_train)

# Model Evaluation
print("Evaluating model...")
y_pred = model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save Model
print("Saving model and vectorizer...")
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Done! model.pkl and vectorizer.pkl created successfully.")
