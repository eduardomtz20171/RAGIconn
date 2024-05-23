import streamlit as st
import requests

st.title("RAG Chatbot with GEMMA and Streamlit")
st.write("Ask a question:")

user_input = st.text_input("Your question:")

if st.button("Get Response"):
    if user_input:
        response = requests.post("http://localhost:8000/chat", json={"query": user_input})
        if response.status_code == 200:
            st.write(response.json()["response"])
        else:
            st.write("Error: Could not get a response from the chatbot.")
    else:
        st.write("Please enter a question.")
