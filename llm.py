import os
import yaml
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import Chroma as DeprecatedChroma  # For backward compatibility

from langchain_chroma import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
import pickle
from langchain import hub


def load_config(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Config file not found: {file_path}")

    with open(file_path, "r") as file:
        try:
            return yaml.safe_load(file)
        except yaml.YAMLError as e:
            raise RuntimeError(f"Error parsing YAML file: {e}")

config = load_config("config.yml")
huggingface_api = config["HUGGING_FACE"]["API_KEY"]

# Set up the Hugging Face Hub API token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_api

# Define the repository ID for the Gemma 2b model
repo_id = "google/gemma-2b-it"

# Set up a Hugging Face Endpoint for Gemma 2b model
llm = HuggingFaceEndpoint(
    repo_id=repo_id, 
    max_length=1024, temperature=0.8,
    timeout=300  # タイムアウトを300秒に延長
)

question = input("Enter your question here:")

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate.from_template(template)
llm_chain = (prompt | llm)

persist_directory = "./chroma_db"

# Recreate the embedding function
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

try:
    # Load the Chroma database
    db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_function,  # Use the correct parameter
    )
except TypeError:
    print("Falling back to the deprecated Chroma class.")
    db = DeprecatedChroma(
        persist_directory=persist_directory,
        embedding_function=embedding_function,
    )

print("Database loaded successfully!")


# Create retriever
retriever = db.as_retriever(search_type="mmr", search_kwargs={'k': 4, 'fetch_k': 20})

# Pull the RAG prompt from LangChain Hub
try:
    prompt = PromptTemplate(
    template=(
        "You are a helpful assistant. Use the following context to answer the question.\n\n"
        "Context:\n{context}\n\n"
        "Question:\n{question}\n\n"
        "Answer with full detail and explanation:"
    )
)

except Exception as e:
    print(f"Error loading RAG prompt: {e}")
    exit()

# Define document formatter
def format_docs(docs):
    if not docs:
        return "No relevant documents found."
    return "\n\n".join(doc.page_content for doc in docs)

# Format the context and question
def format_rag_input(context, question):
    return prompt.format(context=context, question=question)

try:
    rag_chain = (
        RunnablePassthrough()  # Pass the formatted string directly
        | llm  # The HuggingFaceEndpoint processes the input string
    )

    # Retrieve relevant documents
    docs = retriever.invoke(question)
    formatted_context = format_docs(docs)  # Format the documents into a context string

    # Create the input string
    formatted_input = format_rag_input(context=formatted_context, question=question)

    # Run the chain
    rag_result = rag_chain.invoke(formatted_input)
    print(f"RAG Result: {rag_result}")
except Exception as e:
    print(f"Error in RAG Chain: {e}")

