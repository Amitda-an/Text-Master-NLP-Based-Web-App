# NLP Web Applications

This project is a **Flask-based web application** showcasing multiple Natural Language Processing (NLP) functionalities using Python. It provides an interactive web interface where users can analyze and process text data in real-time.  

---

## **Features**

1. **Sentiment Analysis**
   - Classifies user input text as **Positive**, **Negative**, or **Neutral**.
   - Model: Pre-trained machine learning model using scikit-learn.

2. **Spam Detection**
   - Detects whether a given message is **Spam** or **Not Spam (Ham)**.
   - Model: Pre-trained machine learning model using scikit-learn.

3. **Text Summarization**
   - Generates a concise summary of long text using the **LSA (Latent Semantic Analysis) algorithm** from the `sumy` library.

4. **News Classification**
   - Classifies news articles into categories like **Business, Entertainment, Health, Science & Technology**.
   - Model: Pre-trained Logistic Regression model on `20 Newsgroups` dataset.

---

## **Technologies Used**

- **Backend:** Python, Flask
- **NLP Libraries:** spaCy, NLTK, scikit-learn, Sumy
- **Frontend:** HTML, Bootstrap (via Flask templates)
- **Data Serialization:** Joblib (for saving/loading models)

---

## **Project Structure**

