import streamlit as st
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import pickle

# Load the Word2Vec model
word2vec_model = Word2Vec.load('word2vec_model.model')

# Load the XGBoost model
with open('xgb_model.pkl', 'rb') as f:
    xgb_model = pickle.load(f)

# Load the label encoder
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Check the classes in label encoder to ensure they are loaded correctly
print("Label encoder classes:", label_encoder.classes_)

# Preprocessing function (replace with your actual preprocessing function)
def preprocess_text(text):
    # Dummy preprocessing function - replace with your actual preprocessing
    tokens = text.lower().split()
    return tokens

# Function to get average embedding
def get_average_embedding(review_tokens, model):
    embeddings = [model.wv[token] for token in review_tokens if token in model.wv]
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(model.vector_size)

# Streamlit app
st.title('Sentiment Analysis App')

review_text = st.text_area('Enter your review:', '')

if st.button('Predict'):
    if review_text.strip():
        # Preprocess the input
        cleaned_review = preprocess_text(review_text)

        # Extract features
        embedding = get_average_embedding(cleaned_review, word2vec_model)

        # Ensure the embedding is in the correct shape
        embedding = embedding.reshape(1, -1)

        # Make prediction
        prediction = xgb_model.predict(embedding)[0]

        # Debugging: Print the predicted label index
        print("Predicted label index:", prediction)

        # Convert the numerical prediction to the sentiment label
        sentiment_label = label_encoder.inverse_transform([prediction])[0]

        st.write(f'Sentiment: {sentiment_label}')
    else:
        st.write("Please enter a review text.")
