import json
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
# from nltk.chat.util import Chat, reflections
import streamlit as st

import random
import pickle

# load trained model
model = keras.models.load_model("chat_model")

with open("intents.json") as file:
    data = json.load(file)

# load tokenizer object
with open("tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)

# load label encoder object
with open("label_encoder.pickle", "rb") as enc:
    lbl_encoder = pickle.load(enc)

max_len = 20

st.title("Rule based Chatbot")
st.subheader("This is a Rule based Chatbot made using Tensorflow, Keras, NlTK and Python by TK14 ")

def main():
    st.write("Initialize the Chat bot By Typing Hi ")
    inp = st.text_input("Start your chat here")

    result = model.predict(
        keras.preprocessing.sequence.pad_sequences(
            tokenizer.texts_to_sequences([inp]), truncating="post", maxlen=max_len
        )
    )
    tag = lbl_encoder.inverse_transform([np.argmax(result)])

    for i in data["intents"]:
        if i["tag"] == tag:
            st.write("ChatBot: " + np.random.choice(i["responses"]))


if __name__ == "__main__":
    main()
