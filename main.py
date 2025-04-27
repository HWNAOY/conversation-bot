import streamlit as st

from langchain_helper import create_qa_chain, create_vector_db

st.title("Ecommerce ChatBot FAQ(s) 🤖")

btn = st.button("Create Knowledge Base")

if btn:
    create_vector_db()

question = st.text_input("Ask a question: ")

if question:
    chain = create_qa_chain()
    response = chain.invoke(question)

    st.header("Answer: ")
    st.write(response)
