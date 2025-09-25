import streamlit as st
import tensorflow as tf
import numpy as np
import pickle

try:
    model = tf.keras.models.load_model("next_word_model.h5")
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None

try:
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
except Exception as e:
    st.error(f"Error loading tokenizer: {e}")
    tokenizer = None

MAX_SEQUENCE_LEN = 5


def predict_next_words(text, top_n=5):
    if not model or not tokenizer:
        return []
    try:
        sequence = tokenizer.texts_to_sequences([text])[0]
        input_seq = tf.keras.preprocessing.sequence.pad_sequences(
            [sequence], maxlen=MAX_SEQUENCE_LEN
        )
        predictions = model.predict(input_seq, verbose=0)[0]
        top_indices = predictions.argsort()[-top_n:][::-1]
        index_word = {v: k for k, v in tokenizer.word_index.items()}
        top_words = [index_word.get(i, "N/A") for i in top_indices]
        return top_words
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return []


if "user_input" not in st.session_state:
    st.session_state.user_input = ""

st.title("🔤 Smart Next Word Recommender")

user_input = st.text_input(
    "Type your sentence here:", value=st.session_state.user_input
)

if user_input:
    st.session_state.user_input = user_input

    st.write("### Suggested next words:")
    suggestions = predict_next_words(user_input)

    for word in suggestions:
        if st.button(word):
            st.session_state.user_input = st.session_state.user_input + " " + word

            # Instead of using st.experimental_rerun(), use this alternative:
            import _thread

            _thread.interrupt_main()  # Interrupt main thread to rerun Streamlit
