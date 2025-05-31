# sentiment_analysis.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import nltk
from nltk.corpus import stopwords
import string

nltk.download('stopwords')

# 1. Load sample data
data = {
    'text': [
        'I love this product!', 
        'Worst experience ever.', 
        'Not bad, could be better.', 
        'Absolutely fantastic service!', 
        'I will never buy this again.', 
        'It was okay, nothing special.', 
        'Totally worth the money.', 
        'Very disappointed with the quality.'
    ],
    'sentiment': ['positive', 'negative', 'neutral', 'positive', 'negative', 'neutral', 'positive', 'negative']
}

df = pd.DataFrame(data)

# 2. Preprocessing
def clean_text(text):
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = text.split()
    cleaned = [word for word in tokens if word not in stop_words]
    return ' '.join(cleaned)

df['cleaned_text'] = df['text'].apply(clean_text)

# 3. Vectorization
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['cleaned_text'])

# 4. Labels
y = df['sentiment']

# 5. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Model training
model = LogisticRegression()
model.fit(X_train, y_train)

# 7. Predictions and evaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
