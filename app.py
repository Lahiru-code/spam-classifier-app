import streamlit as st
import pickle

# Load saved model and vectorizer
with open("spam_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# App title
st.title("Spam Email Classifier")
st.write("Enter a message below to check whether it is Spam or Not Spam.")

# User input
message = st.text_area("Enter your message:")

# Prediction
if st.button("Predict"):
    if message.strip() == "":
        st.warning("Please enter a message.")
    else:
        message_vec = vectorizer.transform([message])
        prediction = model.predict(message_vec)

        if prediction[0] == 1:
            st.error("This message is Spam")
        else:
            st.success("This message is Not Spam")