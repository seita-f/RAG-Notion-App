# import os
# from dotenv import load_dotenv
# from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
# from langchain_chroma import Chroma
# from langchain.prompts import (
#     PromptTemplate,
#     ChatPromptTemplate,
# )
# from langchain_text_splitters import CharacterTextSplitter
# from langchain_core.runnables import RunnablePassthrough
# from langchain_community.vectorstores import Chroma as DeprecatedChroma
# from langchain.memory import ConversationBufferMemory
# from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain_core.chat_history import BaseChatMessageHistory
# from langchain_community.chat_message_histories import ChatMessageHistory
# from langchain_core.prompts import MessagesPlaceholder

# class RAGApp:
#     def __init__(self, temperature=0.5):
#         # Load environment variables
#         load_dotenv()
#         self.HUGGING_FACE_API_KEY = os.getenv('HUGGING_FACE_API_KEY')

#         if self.HUGGING_FACE_API_KEY:
#             os.environ["HUGGINGFACEHUB_API_TOKEN"] = self.HUGGING_FACE_API_KEY
#             print("Hugging Face API Key set successfully.")
#         else:
#             raise ValueError("HUGGING_FACE_API_KEY is not set in the .env file.")

#         # HuggingFace Endpoint
#         self.repo_id = "google/gemma-2b-it"
#         self.llm = HuggingFaceEndpoint(
#             repo_id=self.repo_id,
#             model_kwargs={"max_length": 1024},  # Set max_length explicitly
#             temperature=temperature, timeout=500
#         )

#         # Embedding function and database
#         self.embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L12-v2")
#         self.db = self.initialize_database()

#         # RAG ChatPromptTemplate
#         template = (
#             "You are a helpful assistant. Use the following context to answer the question.\n\n"
#             "Context:\n{context}\n\n"
#             "Question:\n{question}\n\n"
#             "Answer with full detail and explanation:"
#         )

#         self.prompt = ChatPromptTemplate.from_messages([
#             MessagesPlaceholder(variable_name="history"),
#             ("human", template)
#         ])

#         # Initialize memory
#         self.memory = ConversationBufferMemory(
#             memory_key="history",
#             k=3  # Store last 3 exchanges
#         )

#         self.store = {}

#     def initialize_database(self, persist_directory="./chroma_db"):
#         try:
#             return Chroma(
#                 persist_directory=persist_directory,
#                 embedding_function=self.embedding_function,
#             )
#         except TypeError:
#             print("Falling back to deprecated Chroma class.")
#             return DeprecatedChroma(
#                 persist_directory=persist_directory,
#                 embedding_function=self.embedding_function,
#             )

#     def __format_docs__(self, docs):
#         if not docs:
#             return "No relevant documents found."
#         return "\n\n".join(doc.page_content for doc in docs)

#     def __get_session_history__(self, session_id: str) -> BaseChatMessageHistory:
#         if session_id not in self.store:
#             self.store[session_id] = ChatMessageHistory()
#         return self.store[session_id]

#     def history(self, session_id: str):
#         """Retrieve chat history for a session."""
#         session_history = self.__get_session_history__(session_id)
#         if not session_history.messages:
#             return "No history available."
        
#         # Format messages for display
#         formatted_history = "\n".join(
#             f"{message['role']}: {message['content']}" for message in session_history.messages
#         )
#         return formatted_history

#     def get_response(self, question, session_id):
#         # Retrieve related documents
#         retriever = self.db.as_retriever(search_type="mmr", search_kwargs={'k': 4, 'fetch_k': 20})
#         docs = retriever.invoke(question)
#         formatted_context = self.__format_docs__(docs)

#         # Get session history
#         session_history = self.__get_session_history__(session_id)

#         # Generate formatted prompt input
#         # formatted_input = self.__format_rag_input__(context=formatted_context, question=question, history=session_history.messages)  # Pass the history explicitly)
#         # Generate formatted prompt input
#         formatted_input = self.prompt.format(
#             context=formatted_context,
#             question=question,
#             history=session_history.messages  # Pass the history explicitly
#         )

#         # Interactive conversation chain
#         conversation_chain = RunnableWithMessageHistory(
#             runnable=self.llm,
#             get_session_history=lambda: session_history
#         )

#         # Pass input and history to the chain
#         response = conversation_chain.invoke({
#             "input": formatted_input,
#             "history": session_history.messages
#         })
#         return response

# def main():
#     # Instantiate the RAGApp
#     rag_app = RAGApp()
#     session_id = "default_session"

#     while True:
#         question = input("Enter your question (or 'history' to view past interactions): ")
#         if question.lower() == "history":
#             print("History:")
#             print(rag_app.history())
#         elif question.lower() == "exit":
#             break
#         else:
#             response = rag_app.get_response(question, session_id)
#             print("Answer:")
#             print(response)

# if __name__ == "__main__":
#     main()
