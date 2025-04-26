# E-commerce Chatbot Project

## Overview

This project implements an e-commerce chatbot designed to answer frequently asked questions (FAQs) for a potential e-commerce website. The chatbot aims to provide quick and helpful responses to customer inquiries, enhancing the user experience and reducing the workload on customer support.

<img width="760" alt="image" src="https://github.com/user-attachments/assets/efdfcf37-407a-4f5d-89dd-bce994137c3d" />

## Technologies Used

* **LangChain:** A framework for developing applications powered by language models. It enables the chatbot to understand and respond to user queries in a natural language.
* **FAISS:** A library for efficient similarity search and clustering of dense vectors. It is used to quickly retrieve relevant answers from the knowledge base.
* **Streamlit:** A Python framework for building web applications. It is used to create an interactive user interface for the chatbot.

## Features

* Answers frequently asked questions related to e-commerce.
* Provides information about products, orders, shipping, returns, and other common topics.
* Offers a user-friendly interface for interacting with the chatbot.

## How it Works

1.  **Data Preparation:** The FAQ data is stored in a CSV file (`ecommerce_faq.csv`).
2.  **Embedding Generation:** The questions from the FAQ data are converted into numerical vector representations using a language model.
3.  **Indexing with FAISS:** The generated embeddings are indexed using FAISS for efficient similarity search.
4.  **User Interaction:** Users can interact with the chatbot through a Streamlit web application.
5.  **Query Processing:** The user's question is converted into an embedding vector.
6.  **Answer Retrieval:** FAISS is used to find the most similar question in the indexed data.
7.  **Response Generation:** The chatbot retrieves the answer corresponding to the most similar question and presents it to the user.

## Setup Instructions

1.  **Prerequisites:**
    * Python 3.12
    * pip

2.  **Installation:**

    ```bash
    pip install -r requirements_chatbot.txt
    ```

3.  **Usage:**
    * Clone this repository.
    * Place the `ecommerce_faq.csv' file in the project directory.
    * Run the Streamlit application:

        ```bash
        streamlit run main.py
        ```

    * Interact with the chatbot through the web interface.

## Data Source

The FAQ data is based on common e-commerce customer support questions.
