from flask import Flask, render_template, request
import os
from src.helper import download_huggingface_embedding
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import CTransformers
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import pinecone

app = Flask(__name__)

# Set up Pinecone
PINECONE_API_KEY = "pcsk_4qMWWr_4m4bspViPcAeWzKzaWDVpQg5EZqC5VppiaLZRkhp5aXdCoiQ5VzP7AyNDMbe17m"
os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY
PINECONE_INDEX_NAME = "medical-chatbot"

# Load embeddings
embeddings = download_huggingface_embedding()

# Load Pinecone index
docsearch = PineconeVectorStore.from_existing_index(PINECONE_INDEX_NAME, embeddings)

# Create Prompt
prompt_template = """
Use the given information context to answer the user's question.
If you don't know the answer, say you don't know, but don't make up an answer.

Context: {context}
Question: {question}

Helpful answer:
"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs = {"prompt": PROMPT}

# Load LLM - Optimized for 8GB RAM
config = {
    'max_new_tokens': 256,  # Reduced for memory efficiency
    'temperature': 0.7 # Adjust based on CPU
}

llm = CTransformers(
    model="TheBloke/Llama-2-7B-Chat-GGML",
    model_file="llama-2-7b-chat.ggmlv3.q3_K_S.bin",  # Smaller model file
    model_type="llama",
    config=config
)

# Create Retrieval-based QA
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    print("User input:", msg)
    result = qa.invoke(msg)
    print("Response:", result["result"])
    return str(result["result"])

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
