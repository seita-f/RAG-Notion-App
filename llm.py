import os
import yaml
from dotenv import load_dotenv

# langchain
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough


class RAGApp:

    def __init__(self, embedding_model, llm_model, max_token, temperature=0.4):
        # env
        load_dotenv()
        self.HUGGING_FACE_API_KEY = os.getenv('HUGGING_FACE_API_KEY')

        if self.HUGGING_FACE_API_KEY:
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = self.HUGGING_FACE_API_KEY
            print("Hugging Face API Key set successfully.")
        else:
            raise ValueError("HUGGING_FACE_API_KEY is not set in the .env file.")
        
        # HuggingFace Endpoint
        self.repo_id = llm_model
        self.llm = HuggingFaceEndpoint(
            repo_id=self.repo_id, max_length=max_token, temperature=temperature, timeout=1000
        )

        # Embedding function and database
        self.embedding_function = HuggingFaceEmbeddings(model_name=embedding_model)
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
        except Exception as e:
            print(f"Main loop error: {e}")

    def __format_docs__(self, docs):
        if not docs:
            return "No relevant documents found."
        return "\n\n".join(doc.page_content for doc in docs)

    def get_response(self, question, search_type="mmr", k=4, fetch_k=20, eval_mode=False):

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
        retriever = self.db.as_retriever(search_type=search_type, search_kwargs={'k': k, 'fetch_k': fetch_k})
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

    # load config
    with open('config.yaml') as file:
        config = yaml.safe_load(file.read())
    
    try:
        # Instanace 
        rag_app = RAGApp(config["embedding"]["model"], config["llm"]["model"], config["llm"]["max_token"], config["llm"]["temperature"])

        # User Input
        question = input("Enter your question:")
        response = rag_app.get_response(question, config["llm"]["search_type"], config["llm"]["k"], config["llm"]["fetch_k"], eval_mode=False)
        print("Answer:")
        print(response)

    except Exception as e:
        print(f"Main loop error: {e}")


if __name__ == "__main__":
    main()

