import streamlit as st
import pickle
import re

# Load model and vectorizer
model = pickle.load(open("model/fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("model/tfidf_vectorizer.pkl", "rb"))

# Clean text
def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text.lower()

# Streamlit UI
st.set_page_config(page_title="Fake News Detector", layout="centered")

st.title("📰 Fake News Detector for Students")
st.write("Analyze news articles and verify their credibility using AI.")

user_input = st.text_area("Enter News Article or Headline:")

if st.button("Check News"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        processed_text = clean_text(user_input)
        vectorized_text = vectorizer.transform([processed_text])
        prediction = model.predict(vectorized_text)[0]

        if prediction == 1:
            st.success("✅ This news is REAL")
        else:
            st.error("❌ This news is FAKE")

        st.subheader("📌 News Summary")
        st.write(user_input[:300] + "...")
