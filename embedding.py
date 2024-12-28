import json
import os
import yaml
from dotenv import load_dotenv

# langchain
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document  
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


class JSONHandler:
    @staticmethod  # no need to create class instance 
    def load_json_data(file_path):
        """
        Load data from a JSON file.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def create_documents_from_json(json_data):
        """
        Convert JSON data into document type.
        """
        documents = []
        for key, value in json_data.items():
            if isinstance(value, str):
                documents.append(Document(page_content=value, metadata={"key": key}))
            else:
                print(f"Unexpected data type for key {key}: {type(value)}")
        return documents


class Embedding:
    def __init__(self, embedding_model_name, persist_directory, chunk_size=1024, chunk_overlap=100, separator='\n'):
        self.embedding_model_name = embedding_model_name
        self.persist_directory = persist_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator

        load_dotenv()
        HUGGING_FACE_API_KEY = os.getenv('HUGGING_FACE_API_KEY')

        if HUGGING_FACE_API_KEY:
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGING_FACE_API_KEY
            print("Embedding: Hugging Face API Key set successfully.")
        else:
            raise ValueError("HUGGING_FACE_API_KEY is not set in the .env file.")

    def create_embeddings(self):
        return HuggingFaceEmbeddings(model_name=self.embedding_model_name)

    def clean_text(self, text):
        """
        Clean text data by stripping empty lines.
        """
        return " ".join(line.strip() for line in text.splitlines() if line.strip())

    def split_and_clean_documents(self, documents):
        """
        Splitting the texts into chunks and cleaning.
        """
        text_splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap, separator=self.separator
        )
        split_docs = text_splitter.split_documents(documents)
        cleaned_docs = [
            Document(
                page_content=self.clean_text(doc.page_content),
                metadata=doc.metadata
            )
            for doc in split_docs
        ]
        return cleaned_docs

    def create_and_persist_chroma_db(self, cleaned_docs):
        """
        Create and save Chroma database.
        """
        embedding_function = self.create_embeddings()

        if not os.path.exists(self.persist_directory):
            os.makedirs(self.persist_directory)

        db = Chroma.from_documents(
            cleaned_docs,
            embedding_function,
            persist_directory=self.persist_directory,
        )
        print(f"Database saved successfully to disk at {self.persist_directory}")


def main():

    # load config
    with open('config.yaml') as file:
        config = yaml.safe_load(file.read())

    try:
        # Load and process the JSON data
        json_data = JSONHandler.load_json_data(config["notion"]["content_file"])
        documents = JSONHandler.create_documents_from_json(json_data)

        # Create EmbeddingHandler instance and process the documents
        embedding_handler = Embedding(
            embedding_model_name=config["embedding"]["model"],
            persist_directory=config["embedding"]["db_dir"],
            chunk_size=config["embedding"]["chunk_size"],
            chunk_overlap=config["embedding"]["overlap"],
        )

        cleaned_docs = embedding_handler.split_and_clean_documents(documents)
        embedding_handler.create_and_persist_chroma_db(cleaned_docs)

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
