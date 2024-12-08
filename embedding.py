import json
import os
from langchain_text_splitters import CharacterTextSplitter
from langchain.docstore.document import Document  # 正しいインポートパス
from langchain.schema import Document  # LangChainのDocumentクラスを使用する場合
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# JSONファイルを読み込む
with open("notion-api/notion_contents.json", "r", encoding="utf-8") as f:
    json_data = json.load(f)


# JSONの内容を1つのテキストに結合する
data = []
for key, value in json_data.items():
    # 値が文字列である場合のみ処理
    if isinstance(value, str):
        # エスケープされた文字列をデコード
        normalized_content = value.replace("\\n", "\n").strip()
        # Documentオブジェクトに変換
        data.append(Document(page_content=f"{key}: {normalized_content}"))
    else:
        print(f"Unexpected data type for key {key}: {type(value)}")

# 結果を表示
for doc in data:
    print(f"Document content: {doc.page_content}")

# テキストを分割
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(data)

# 埋め込み関数を作成
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Chroma データベースの永続化ディレクトリを指定
persist_directory = "./chroma_db"

if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

def clean_text(text):
    return " ".join(line.strip() for line in text.splitlines() if line.strip())

# ドキュメントをクリーンアップ
cleaned_docs = [
    Document(page_content=clean_text(doc.page_content)) for doc in docs
]

# Chroma データベースを作成し永続化
db = Chroma.from_documents(
    cleaned_docs,
    embedding_function,
    persist_directory=persist_directory,
)

# 永続化処理
db.persist()

print(f"Database saved successfully to disk at {persist_directory}")
