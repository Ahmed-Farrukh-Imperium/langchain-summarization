import os
from utils.env_loader import load_env_vars
from utils.summarizer import summarizer_chain
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Load env
load_env_vars()

embedding_model = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")


# Load text file
loader = TextLoader("ai_intro.txt")
docs = loader.load()

# Split into chunks
splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = splitter.split_documents(docs)

# Build embeddings + vectorstore
embeddings = AzureOpenAIEmbeddings(
    model=embedding_model
)

vectorstore = FAISS.from_documents(chunks, embeddings)

# Create retriever
retriever = vectorstore.as_retriever()

# Retrieve docs for query
query = "AI milestones"
retrieved_docs = retriever.get_relevant_documents(query)

# Combine retrieved chunks into one string
retrieved_text = " ".join([doc.page_content for doc in retrieved_docs])

summarizer_chain(retrieved_text)

