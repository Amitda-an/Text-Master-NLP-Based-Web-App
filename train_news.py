from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
data = fetch_20newsgroups(subset='all', remove=('headers','footers','quotes'))

X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

pipeline = Pipeline([
    ('vectorizer', CountVectorizer(stop_words='english', max_features=10000)),
    ('clf', LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)
print("News Accuracy:", pipeline.score(X_test, y_test))

joblib.dump(pipeline, "models/news_classifier.pkl")
print("✅ News model saved to models/news_classifier.pkl")
