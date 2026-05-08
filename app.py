import streamlit as st
import pickle
import re

# Load the saved model and vectorizer
@st.cache_resource
def load_models():
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    return model, vectorizer

try:
    model, vectorizer = load_models()
except FileNotFoundError:
    st.error("Model files not found. Please run 'python train_model.py' first.")
    st.stop()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text

def predict(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned]).toarray()
    prediction = model.predict(vec)[0]
    return prediction

st.title("🧠 Sentiment Analysis App")
st.write("Enter some text below to find out if the sentiment is Positive or Negative!")

user_input = st.text_area("Enter text here:", height=100)

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text first.")
    else:
        result = predict(user_input)
        
        if result.lower() == "positive":
            st.success(f"Sentiment: **{result}** 🥳")
        else:
            st.error(f"Sentiment: **{result}** 😡")
