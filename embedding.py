import json
from langchain_text_splitters import CharacterTextSplitter
from langchain.docstore.document import Document  # 正しいインポートパス
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# JSONファイルを読み込む
with open("notion-api/notion_contents.json", "r", encoding="utf-8") as f:
    json_data = json.load(f)

# JSONの内容を1つのテキストに結合する
data = []
for key, values in json_data.items():
    for value in values:
        # Documentオブジェクトに変換
        data.append(Document(page_content=f"{key}: {value}"))

# テキストを分割
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(data)

# 埋め込み関数を作成
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Chroma データベースの永続化ディレクトリを指定
persist_directory = "./chroma_db"

# Chroma データベースを作成し永続化
db = Chroma.from_documents(
    docs,
    embedding_function,
    persist_directory=persist_directory,
)

# 永続化処理
db.persist()

print(f"Database saved successfully to disk at {persist_directory}")
