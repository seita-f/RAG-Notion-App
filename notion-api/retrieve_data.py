import requests
import yaml
import json
import unicodedata
import os
import sys


def load_config(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Config file not found: {file_path}")

    with open(file_path, "r") as file:
        try:
            return yaml.safe_load(file)
        except yaml.YAMLError as e:
            raise RuntimeError(f"Error parsing YAML file: {e}")

# Notionページコンテンツを取得（ページング対応）
def get_all_blocks(page_id, headers, notion_api_url):
    all_blocks = []
    url = f"{notion_api_url}/{page_id}/children"
    while url:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            all_blocks.extend(data.get("results", []))
            # ページング処理
            if data.get("has_more"):
                url = f"{notion_api_url}/{page_id}/children?start_cursor={data.get('next_cursor')}"
            else:
                url = None
        else:
            print(f"Error: {response.status_code}, {response.text}")
            break
    return all_blocks

def normalize_text_data(content_list):
    normalized_content_list = [unicodedata.normalize('NFKC', text).strip().lower() for text in content_list]   
    return normalized_content_list 

# ページタイトルを取得
def get_page_title(page_id, headers):
    url = f"https://api.notion.com/v1/pages/{page_id}"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        page_data = response.json()
        properties = page_data.get("properties", {})
        for prop in properties.values():
            if prop.get("type") == "title":
                title_texts = prop.get("title", [])
                return "".join([text["plain_text"] for text in title_texts])
        return "Untitled"
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None


# ページ内のテキストを抽出（修正版）
def extract_text_from_blocks(blocks, headers, notion_api_url):
    content_list = []
    for block in blocks:
        block_type = block.get("type")
        
        # テキスト系ブロック
        if block_type in ["paragraph", "heading_1", "heading_2", "heading_3", "bulleted_list_item", "numbered_list_item"]:
            rich_texts = block[block_type].get("rich_text", [])
            content_list.extend([text["text"]["content"] for text in rich_texts if "text" in text])
        
        # テーブルのブロック
        elif block_type == "table":
            table_id = block["id"]
            child_blocks = get_all_blocks(table_id, headers, notion_api_url)
            for row in child_blocks:
                if row.get("type") == "table_row":
                    row_cells = row["table_row"]["cells"]
                    row_text = ["".join([cell["text"]["content"] for cell in cell_texts if "text" in cell]) for cell_texts in row_cells]
                    content_list.append("\t".join(row_text))  # セルをタブ区切りで結合
        
        # 子ブロックの再帰処理
        if block.get("has_children"):
            child_id = block["id"]
            child_blocks = get_all_blocks(child_id, headers, notion_api_url)
            content_list.extend(extract_text_from_blocks(child_blocks, headers, notion_api_url))
    
    return content_list

# メイン処理（修正版）
def main():
    # YAML設定ファイルを読み込む
    config = load_config("../config.yml")
    NOTION_API_KEY = config["NOTION"]["API_KEY"]
    NOTION_API_URL = config["NOTION"]["API_URL"]
    NOTION_VERSION = config["NOTION"]["VERSION"]
    PAGE_IDS = config["NOTION"]["PAGE_IDS"]

    # ヘッダー設定
    headers = {
        "Authorization": f"Bearer {NOTION_API_KEY}",
        "Notion-Version": NOTION_VERSION
    }

    # ページごとにコンテンツを取得
    all_content = {}
    for page_id in PAGE_IDS:
        # ページのタイトルを取得
        title = get_page_title(page_id, headers)

        # ページのすべてのブロックを取得
        blocks = get_all_blocks(page_id, headers, NOTION_API_URL)

        # ブロックからテキストを抽出
        page_content = extract_text_from_blocks(blocks, headers, NOTION_API_URL)
        normalized_page_content = normalize_text_data(page_content)

        # コンテンツを1つの文字列に結合
        full_text = "\n".join(normalized_page_content)

        # タイトルをキーにして保存
        all_content[title] = full_text
        print(f"Retrieved the contents from {title}")

    # 取得結果をJSON形式で保存
    with open("notion_contents.json", "w", encoding="utf-8") as file:
        json.dump(all_content, file, ensure_ascii=False, indent=4)

    print("Content saved to notion_contents.json")

# 実行
if __name__ == "__main__":
    main()