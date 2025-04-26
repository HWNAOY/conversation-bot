import os
import pickle
import faiss
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain import hub

load_dotenv()

filename = "ecommerce_faq.csv"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0)

hugging_face_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
)

def create_vector_db():
    # Load documents from CSV
    loader = CSVLoader(file_path=filename, source_column="prompt")
    docs = loader.load()
    faiss_index = FAISS.from_documents(
        documents=docs,
        embedding=hugging_face_embeddings
    )
    faiss.write_index(faiss_index.index, "faiss_index.bin")

    # Save the docstore and index_to_docstore_id
    with open("faiss_index_docstore.pkl", "wb") as f:
        pickle.dump(faiss_index.docstore, f)

    with open("faiss_index_index_to_docstore_id.pkl", "wb") as f:
        pickle.dump(faiss_index.index_to_docstore_id, f)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def create_qa_chain():
    # Load the vector store
    # vector_db = FAISS.load_local("faiss_index", hugging_face_embeddings)
    loaded_index = faiss.read_index("faiss_index.bin")
    # Load the docstore and index_to_docstore_id
    with open("faiss_index_docstore.pkl", "rb") as f:
        doc_store = pickle.load(f)

    with open("faiss_index_index_to_docstore_id.pkl", "rb") as f:
        index_to_docstore_id = pickle.load(f)
  
    vector_db = FAISS(
        index=loaded_index,
        embedding_function=hugging_face_embeddings,
        docstore=doc_store,
        index_to_docstore_id=index_to_docstore_id
    )
    retriever = vector_db.as_retriever()
    # Create the QA chain
    # prompt_template = """Answer the following based as much as possible 
    #                     from the provided context: {context}. If answer not
    #                     found in the context, state "I don't know".

    #                     Question: {question}"""
    # prompt_template = """You are an expert assistant. 
    #                     Answer the question strictly based on the provided context below.

    #                     <context>
    #                     {context}
    #                     </context>

    #                     - If the answer is stated in the context, copy the relevant sentence(s) directly from context.
    #                     - DO NOT summarize in your own words.
    #                     - If the answer is not in the context, reply exactly: "I don't know."
    #                     - Never answer just "Yes" or "No" unless those words appear inside a full sentence in context.
    #                     - Keep the answer short, factual, and copied as-is from the context.

    #                     Question: {question}

    #                     Answer:"""
    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    prompt = ChatPromptTemplate.from_template(prompt_template)
    # prompt = hub.pull("rlm/rag-prompt")

    qa_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return qa_chain

if __name__ == "__main__":
    # create_vector_db()
    chain = create_qa_chain()
    query = "How to create an account?"
    response = chain.invoke(query)
    print(response)
