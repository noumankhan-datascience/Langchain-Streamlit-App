import os
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain

# Set environment variables
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) LangChainBot"
os.environ["GOOGLE_API_KEY"] = "AIzaSyDhGg8nBMVzMr35LO9x9bEgoscTN422VhQ"  # ← Replace with your real key

# Load and split website content
loader = WebBaseLoader("https://datacrumbs.org/")
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash-latest",  # or use "models/chat-bison-001" if needed
    temperature=0.7,
)

# Load QA chain
chain = load_qa_chain(llm, chain_type="stuff")

# Streamlit UI
st.title("Chat with Datacrumbs Website")
question = st.text_input("Ask a question about the Datacrumbs website or anything else:")

if question:
    response = chain.run(input_documents=docs, question=question)
    st.write(response)
