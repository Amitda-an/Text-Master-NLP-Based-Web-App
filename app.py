from flask import Flask, render_template, request, url_for

# nlp library
import spacy
import joblib
from spacy.lang.en.stop_words import STOP_WORDS
import string

# summarization (using sumy instead of gensim)
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

# load the spacy english model (new way)
nlp = spacy.load('en_core_web_sm')
punct = string.punctuation
stopwords = list(STOP_WORDS)

# model load sentiment analysis:
sen_model = joblib.load('models/sentiment.pkl')

# model load spam classification
spam_model = joblib.load('models/spam.pkl')

# news classifier
n_clf = joblib.load('models/news_classifier.pkl')

app = Flask(__name__)

# home page
@app.route('/')
def index():
    return render_template('home.html')

# sentiment analysis
@app.route('/nlpsentiment')
def sentiment_nlp():
    return render_template('sentiment.html')

@app.route('/sentiment', methods=['POST', 'GET'])
def sentiment():
    if request.method == 'POST':
        message = request.form['message']
        pred = sen_model.predict([message])
        return render_template('sentiment.html', prediction=pred)

# spam
@app.route('/nlpspam')
def spam_nlp():
    return render_template('spam.html')

@app.route('/spam', methods=['POST', 'GET'])
def spam():
    if request.method == 'POST':
        message = request.form['message']
        pred = spam_model.predict([message])
        return render_template('spam.html', prediction=pred)

# summarization
@app.route('/nlpsummarize')
def summarize_nlp():
    return render_template('summarize.html')

@app.route('/summarize', methods=['POST', 'GET'])
def sum_route():
    if request.method == 'POST':
        message = request.form['message']
        # sumy summarizer
        parser = PlaintextParser.from_string(message, Tokenizer("english"))
        summarizer = LsaSummarizer()
        summary_sentences = summarizer(parser.document, 2)  # 2 = number of sentences
        sum_message = " ".join([str(s) for s in summary_sentences])
        return render_template('summarize.html', original=message, prediction=sum_message)

# news classifier
@app.route('/newsclf')
def news_classifier():
    return render_template('news.html')

@app.route('/newsclassifier', methods=['POST', 'GET'])
def news_clf_route():
    if request.method == 'POST':
        message = request.form['message']
        pred = n_clf.predict([message])[0]
        category = n_clf.classes_[pred]
        return render_template('news.html', prediction=category)
    return render_template('news.html')


# ✅ Add this to actually run Flask
if __name__ == "__main__":
    app.run(debug=True)