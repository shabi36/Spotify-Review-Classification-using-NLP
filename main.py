import pickle
import streamlit as st
import nltk
import string

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

# Load the vectorizer and model
with open("vectorizer_spotify_reviews.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("model_mnb_spotify_reviews.pkl", "rb") as f:
    model = pickle.load(f)

nltk.download("punkt")
nltk.download("stopwords")


st.title("SPOTIFY REVIEWS  ANALYSIS")

input = st.text_area("Enter The Review")


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y.copy()
    y.clear()

    for i in text:
        if i not in stopwords.words("english") and i not in string.punctuation:
            y.append(i)

    text = y.copy()
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


if st.button("Analyse"):
    # 1) Preprocess
    transformed_input = transform_text(input)

    # 2) Vectorize
    vector_input = tfidf.transform([transformed_input])

    # 3) Predict
    result = model.predict(vector_input)[0]

    # 4) Display Result
    if result == 0:
        st.header("Negative Review")
    if result == 1:
        st.header("Neutral Review")
    else:
        st.header("Positive Review")