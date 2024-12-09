import json
import os
from langchain_text_splitters import CharacterTextSplitter
from langchain.schema import Document  # LangChainのDocumentクラスを使用
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ディレクトリやファイルパスの設定
PERSIST_DIRECTORY_PATH = "./chroma_db"
CONTENTS_JSON_PATH = "notion-api/notion_contents.json"


def load_json_data(file_path):
    """JSONファイルを読み込む"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_documents_from_json(json_data):
    """JSONデータからDocumentオブジェクトを作成"""
    documents = []
    for key, value in json_data.items():
        if isinstance(value, str):
            # エスケープされた文字列を正規化
            normalized_content = value.replace("\\n", "\n").strip()
            # Documentオブジェクトを作成
            documents.append(Document(page_content=f"{key}: {normalized_content}"))
        else:
            print(f"Unexpected data type for key {key}: {type(value)}")
    return documents


def clean_text(text):
    """テキストをクリーンアップ"""
    return " ".join(line.strip() for line in text.splitlines() if line.strip())


def split_and_clean_documents(documents, chunk_size=1000, chunk_overlap=0):
    """ドキュメントを分割してクリーンアップ"""
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = text_splitter.split_documents(documents)
    cleaned_docs = [Document(page_content=clean_text(doc.page_content)) for doc in split_docs]
    return cleaned_docs


def create_and_persist_chroma_db(cleaned_docs, embedding_model_name, persist_directory):
    """Chromaデータベースを作成して永続化"""
    # 埋め込み関数を作成
    embedding_function = HuggingFaceEmbeddings(model_name=embedding_model_name)
    
    # ディレクトリを作成（存在しない場合）
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)
    
    # Chroma データベースを作成
    db = Chroma.from_documents(
        cleaned_docs,
        embedding_function,
        persist_directory=persist_directory,
    )
    # db.persist()
    print(f"Database saved successfully to disk at {persist_directory}")


def main():
    try:
        # JSONファイルを読み込む
        json_data = load_json_data(CONTENTS_JSON_PATH)
        
        # JSONデータからDocumentを作成
        documents = create_documents_from_json(json_data)

        # DEBUG:
        # for doc in documents:
        #     print(f"Document content: {doc.page_content}")
        
        # ドキュメントを分割してクリーンアップ
        cleaned_docs = split_and_clean_documents(documents)
        
        # Chromaデータベースを作成して永続化
        create_and_persist_chroma_db(
            cleaned_docs,
            embedding_model_name="all-MiniLM-L6-v2",
            persist_directory=PERSIST_DIRECTORY_PATH,
        )
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
