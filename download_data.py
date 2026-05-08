import pandas as pd
import nltk
from nltk.corpus import movie_reviews
import ssl

# Fix potential SSL error for NLTK download on Windows
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

print("Downloading NLTK movie reviews dataset (this might take a few seconds)...")
nltk.download('movie_reviews')

print("Processing reviews...")
documents = []
for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        # We replace newlines with spaces to keep CSV clean
        text = movie_reviews.raw(fileid).replace('\n', ' ')
        documents.append((text, category))

df = pd.DataFrame(documents, columns=['Text', 'Sentiment'])
df['Sentiment'] = df['Sentiment'].map({'pos': 'Positive', 'neg': 'Negative'})

# Shuffle the data
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df.to_csv('data.csv', index=False)
print(f"Success! Created data.csv with {len(df)} real movie reviews.")
