# Step 1: Import Libraries and Load the Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st



# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# @st.cache_data()
# def load_model_data():
#     # Load the IMDB dataset
#     model = load_model("simple_rnn_imdb.h5")
#     return model

# model = load_model_data()

model = load_model("simple_rnn_imdb.h5")

# Step 2: Helper Functions
# Function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review


## Streamlit App

st.set_page_config(
    page_title="Movie Review Sentiment analysis", page_icon="🎇", layout="wide"
)

st.title("🎇 Movie Review Sentiment analysis")

tab1,tab2 = st.tabs(["Model Info","Let's Analyze"])

with tab2:

    st.write("The is a '1%' chance that the model may predict wrong. If you get a review which falls under that category, You are Lucky!! 💯.")

    st.write("Write a review to classify it as Positive or Negative")


    user_input = st.text_area('Movie Review')

    if st.button("Predict"):

        preprocessed_input = preprocess_text(user_input)

        prediction = model.predict(preprocessed_input)

        sentiment = 'Positive' if prediction[0][0] >0.5 else 'Negative'

        if sentiment == 'Positive':
            st.success(f'Sentiment: {sentiment} ☑️')
        else:
            st.error(f'Sentiment: {sentiment} ❌')
        st.write(f'The Prediction score: {prediction[0][0]} ')

    else:

        st.write("Please enter a review and click the button to predict the sentiment")

with tab1:
    st.subheader("| Tips to write a review:")
    st.write("1. Do not add any emoji's in the review, because our model is not trained with that 😅.")
    st.write("2. Please use English to write a review, because our model is trained with English data only 🤖.")
    st.write("3. Don't write regional languages using english like 'Chindi itu' or 'kirak undi' or 'Mast tha' etc")
    st.write("4. Don't use hash tags as well.")
    st.subheader("| Note:")
    st.write("If the code breaks there are two options")
    st.write("1. If you are a coder you can go to my Github and fork the code to make the changes. If it works please email me the changes 😁")
    st.write("2. If you are a non-coder then please refresh the page and try again 🤗")

    st.subheader("| Github")
    st.write(
        '<p>To check out the code, Visit <a href="https://github.com/Koushik-j/Sentimental_Analysis-Movie_Review-Using-RNN">GitHub</a>.</p><br>',
        unsafe_allow_html=True,
    )

st.markdown("---")




