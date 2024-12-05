import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from rag_utils.utils import load_config
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

config = load_config("config.yml")
huggingface_api = config["HUGGING_FACE"]["API_KEY"]

# Set up the Hugging Face Hub API token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_api

# Define the repository ID for the Gemma 2b model
repo_id = "google/gemma-2b-it"

# Set up a Hugging Face Endpoint for Gemma 2b model
llm = HuggingFaceEndpoint(
    repo_id=repo_id, 
    max_length=1024, temperature=0.3,
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
    # prompt = hub.pull("rlm/rag-prompt")
    # Replace hub.pull with a simple test prompt
    prompt = PromptTemplate(template="Context: {context}\n\nQuestion: {question}\n\nAnswer:")
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


# from langchain import hub
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough
# from langchain.chains import RetrievalQA
# import pickle

# # 保存したChromaデータベースを読み込む
# with open("chroma_db.pkl", "rb") as f:
#     db = pickle.load(f)

# retriever = db.as_retriever(search_type="mmr", search_kwargs={'k': 4, 'fetch_k': 20})
# prompt = hub.pull("rlm/rag-prompt")

# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)

# rag_chain = (
#     {"context": retriever | format_docs, "question": RunnablePassthrough()}
#     | prompt
#     | llm
# )

# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain

# # Create a conversation buffer memory
# memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

# # Define a custom template for the question prompt
# custom_template = """Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question, in its original English.
#                         Chat History:
#                         {chat_history}
#                         Follow-Up Input: {question}
#                         Standalone question:"""

# # Create a PromptTemplate from the custom template
# CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)

# # Create a ConversationalRetrievalChain from an LLM with the specified components
# conversational_chain = ConversationalRetrievalChain.from_llm(
#     llm=llm,
#     chain_type="stuff",
#     retriever=db.as_retriever(),
#     memory=memory,
#     condense_question_prompt=CUSTOM_QUESTION_PROMPT
# )