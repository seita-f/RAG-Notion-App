import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_chroma import Chroma 
from langchain.prompts import PromptTemplate

from langchain_text_splitters import CharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import Chroma as DeprecatedChroma


class RAGApp:
    def __init__(self, temperature=0.4):
        # env
        load_dotenv()
        self.HUGGING_FACE_API_KEY = os.getenv('HUGGING_FACE_API_KEY')

        if self.HUGGING_FACE_API_KEY:
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = self.HUGGING_FACE_API_KEY
            print("Hugging Face API Key set successfully.")
        else:
            raise ValueError("HUGGING_FACE_API_KEY is not set in the .env file.")
        
        # HuggingFace Endpoint
        self.repo_id = "google/gemma-2b-it"
        self.llm = HuggingFaceEndpoint(
            repo_id=self.repo_id, max_length=1024, temperature=temperature, timeout=500
        )

        # Embedding function and database
        # self.embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.embedding_function = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
        self.db = self.initialize_database()

        # prompt
        self.prompt = PromptTemplate(
            template=(
                "You are an expert assistant. Use the following context to answer the question.\n"
                "Context:\n{context}\n"
                "Question:\n{question}\n"
                # "Provide a detailed response based only on the given context."
            )
        )

        # prompt for conversation
        # self.prompt = ChatPromptTemplate.from_messages([
        #     ("system", "You are a helpful assistant. Use the following context to answer the question."),
        #     ("human", "Context: {context}. Question: {question}. Please answer with full detail and explanation:")
        # ])
    
    def initialize_database(self, persist_directory="./chroma_db"):
        try:
            return Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embedding_function,
            )
        except TypeError:
            print("Falling back to deprecated Chroma class.")
            return DeprecatedChroma(
                persist_directory=persist_directory,
                embedding_function=self.embedding_function,
            )

    def __format_docs__(self, docs):
        if not docs:
            return "No relevant documents found."
        return "\n\n".join(doc.page_content for doc in docs)

    def get_response(self, question, eval_mode=False):

        #-------- Test: raw answer (no RAG system) ----
        # template = """Question: {question}
        # Answer: Answer with full detail and explanation"""
        # prompt = PromptTemplate.from_template(template)
        # llm_chain = (
        #     RunnablePassthrough() 
        #     | self.llm
        # )
        # result = llm_chain.invoke(question)
        #----------------------------------
        
        rag_chain = (
            RunnablePassthrough()  # Pass the formatted string directly
            | self.llm  # The HuggingFaceEndpoint processes the input string
        )

        # search type: similarity, mmr, ...
        retriever = self.db.as_retriever(search_type="mmr", search_kwargs={'k': 4, 'fetch_k': 20})
        docs = retriever.invoke(question)

        if eval_mode:

            # only evalaute top context due to my PC spec
            formatted_context = [docs[0].page_content] if docs[0] else ["No relevant documents found."]
            
            # Format the input
            formatted_input = self.prompt.format(context=formatted_context, question=question)
            # print("========== DEBUG: Formatted Input ==========")
            # print(formatted_input)

            # Invoke the chain and debug the output
            result = rag_chain.invoke(formatted_input)
            # print("========== DEBUG: Raw LLM Output ==========")
            # print(result)

            return {
                "user_input": question,
                "contexts": formatted_context,
                "response": result.strip(),  # Strip unnecessary whitespace
            }
        else:
            formatted_context = self.__format_docs__(docs)  # Format the documents into a context string
            # generate prompt
            formatted_input = self.prompt.format(context=formatted_context, question=question)

            print("DEBUG:")
            print(formatted_input)

            result = rag_chain.invoke(formatted_input)

            if "The context explains that" in result:
                cleaned_response = result.replace("The context explains that ", "")
                return cleaned_response

            return result

def main():
    # インスタンス作成
    rag_app = RAGApp()

    # ユーザー入力
    question = input("Enter your question:")
    response = rag_app.get_response(question)
    print("Answer:")
    print(response)


if __name__ == "__main__":
    main()

