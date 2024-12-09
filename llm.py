import os
import yaml
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough

class RAGApp:
    def __init__(self, config_path="config.yml", temperature=0.5):
        # Load config
        self.config = self.load_config(config_path)
        self.huggingface_api = self.config["HUGGING_FACE"]["API_KEY"]
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = self.huggingface_api

        # HuggingFace Endpoint setup
        self.repo_id = "google/gemma-2b-it"
        self.llm = HuggingFaceEndpoint(
            repo_id=self.repo_id, max_length=1024, temperature=temperature, timeout=500
        )

        # Embedding function and database
        self.embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.db = self.initialize_database()

        # RAG prompt
        self.prompt = PromptTemplate(
            template=(
                "You are a helpful assistant. Use the following context to answer the question.\n\n"
                "Context:\n{context}\n\n"
                "Question:\n{question}\n\n"
                "Answer with full detail and explanation:"
            )
        )

    @staticmethod
    def load_config(file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Config file not found: {file_path}")

        with open(file_path, "r") as file:
            try:
                return yaml.safe_load(file)
            except yaml.YAMLError as e:
                raise RuntimeError(f"Error parsing YAML file: {e}")

    def initialize_database(self, persist_directory="./chroma_db"):
        try:
            return Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embedding_function,
            )
        except TypeError:
            print("Falling back to deprecated Chroma class.")
            from langchain_community.vectorstores import Chroma as DeprecatedChroma
            return DeprecatedChroma(
                persist_directory=persist_directory,
                embedding_function=self.embedding_function,
            )

    # Define document formatter
    def __format_docs__(self, docs):
        if not docs:
            return "No relevant documents found."
        return "\n\n".join(doc.page_content for doc in docs)

    # Format the context and question
    def __format_rag_input__(self, context, question):
        return self.prompt.format(context=context, question=question)

    def get_response(self, question):

        #-------- For testing (no RAG) ----
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

        # Retrieve relevant documents
        retriever = self.db.as_retriever(search_type="mmr", search_kwargs={'k': 4, 'fetch_k': 20})
        docs = retriever.invoke(question)
        formatted_context = self.__format_docs__(docs)  # Format the documents into a context string

        # Create the input string
        formatted_input = self.__format_rag_input__(context=formatted_context, question=question)

        # Run the chain
        result = rag_chain.invoke(formatted_input)

        # Remove unnecessary comment
        if "The context explains that" in result:
            cleaned_response = result.replace("The context explains that ", "")
            return cleaned_response

        return result

def main():
    # Initialize RAGApp
    rag_app = RAGApp()

    # User input
    question = input("Enter your question:")
    response = rag_app.get_response(question)
    print("Answer:")
    print(response)


if __name__ == "__main__":
    main()

