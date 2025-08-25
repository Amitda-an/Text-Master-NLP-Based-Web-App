import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

# Load dataset
data = pd.read_csv("data/IMDB Dataset.csv")
data['sentiment'] = data['sentiment'].map({'positive':1, 'negative':0})

X_train, X_test, y_train, y_test = train_test_split(
    data['review'], data['sentiment'], test_size=0.2, random_state=42
)

# Pipeline (vectorizer + model)
pipeline = Pipeline([
    ('vectorizer', CountVectorizer(stop_words='english', max_features=5000)),
    ('clf', LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)
print("Sentiment Accuracy:", pipeline.score(X_test, y_test))

joblib.dump(pipeline, "models/sentiment.pkl")
print("✅ Sentiment model saved to models/sentiment.pkl")
