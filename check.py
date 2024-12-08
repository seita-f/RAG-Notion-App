# from langchain_community.document_loaders import WebBaseLoader

# import os
# from langchain_huggingface import HuggingFaceEndpoint
# from langchain_huggingface import HuggingFaceEmbeddings

# from langchain.chains import LLMChain
# from langchain.prompts import PromptTemplate
# from rag_utils.utils import load_config
# from langchain_community.document_loaders import TextLoader
# from langchain_community.embeddings.sentence_transformer import (
#     SentenceTransformerEmbeddings,
# )
# from langchain_community.vectorstores import Chroma
# from langchain_community.vectorstores import Chroma as DeprecatedChroma  # For backward compatibility

# from langchain_chroma import Chroma
# from langchain_text_splitters import CharacterTextSplitter
# from langchain_core.runnables import RunnablePassthrough
# import pickle
# from langchain import hub


# config = load_config("config.yml")
# huggingface_api = config["HUGGING_FACE"]["API_KEY"]

# # Set up the Hugging Face Hub API token
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_api

# # Define the repository ID for the Gemma 2b model
# repo_id = "google/gemma-2b-it"

# # Set up a Hugging Face Endpoint for Gemma 2b model
# llm = HuggingFaceEndpoint(
#     repo_id=repo_id, 
#     max_length=1024, temperature=0.8,
#     timeout=300  # タイムアウトを300秒に延長
# )

# loader = WebBaseLoader("https://www.connectome.design/consulting")
# data = loader.load()
# print()
# print(data[:10])
# print()

# # split it into chunks
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# docs = text_splitter.split_documents(data)

# # create the open-source embedding function
# embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# # load it into Chroma
# db = Chroma.from_documents(docs, embedding_function)

# from langchain.chains import RetrievalQA

# retriever = db.as_retriever(search_type="mmr", search_kwargs={'k': 4, 'fetch_k': 20})
# prompt = hub.pull("rlm/rag-prompt")

# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)

# rag_chain = (
#     {"context": retriever | format_docs, "question": RunnablePassthrough()}
#     | prompt
#     | llm
# )

# question = input("Please enter your question here: ")
# result = rag_chain.invoke(question)
# print(result)
