import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

# Load dataset
data = pd.read_csv("data/spam.csv", encoding='latin-1')[['v1','v2']]
data.columns = ['label','message']
data['label'] = data['label'].map({'ham':0, 'spam':1})

X_train, X_test, y_train, y_test = train_test_split(
    data['message'], data['label'], test_size=0.2, random_state=42
)

pipeline = Pipeline([
    ('vectorizer', CountVectorizer(stop_words='english')),
    ('clf', LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)
print("Spam Accuracy:", pipeline.score(X_test, y_test))

joblib.dump(pipeline, "models/spam.pkl")
print("✅ Spam model saved to models/spam.pkl")
