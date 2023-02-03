import streamlit as st
import nltk
import pickle
from nltk.corpus import stopwords
import re
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
model = pickle.load(open('model2.pkl', 'rb'))
vectorizer = pickle.load(open('tfidfvect2.pkl', 'rb'))

def predict(text):
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    review_vect = vectorizer.transform([review]).toarray()
    prediction = 'THIS IS REAL NEWS' if model.predict(review_vect) == 0 else 'THIS IS FAKE NEWS'
    return prediction

def main():
    st.title("Fake News Classifier")
    text = st.text_input("Enter the text:")
    if text:
        prediction = predict(text)
        st.write("Prediction: ", prediction)

if __name__ == '__main__':
    main()