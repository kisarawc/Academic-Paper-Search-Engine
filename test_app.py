import streamlit as st

st.title("Test Streamlit App")

query = st.text_input("Enter your search query:")

if query:
    st.write(f"You entered: {query}")
