"""
UI using streamlit
"""
import sys
import os
import streamlit as st
from streamlit_option_menu import option_menu  # サイドメニューのライブラリ

# プロジェクトのルートディレクトリをモジュール検索パスに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llm import RAGApp

# ユーザーインターフェースの設定
st.set_page_config(page_title="RAG Desktop App", layout="wide")

# サイドバーのアイコンメニュー
with st.sidebar:
    selected = option_menu(
        menu_title="Menu",
        options=["Chat", "Settings"],
        icons=["chat", "gear"],  # アイコンはFontAwesomeを使用
        menu_icon="menu-down",
        default_index=0,
    )

# Chatが選択された場合のレイアウト
if selected == "Chat":
    st.title("Chat with RAG Bot")
    
    # 入力フォーム
    user_input = st.text_input("Enter your message:")
    if st.button("Send"):
        if user_input:
            # RAGApp を使用して応答を生成
            rag_app = RAGApp()
            response = rag_app.get_response(user_input)

            # ボットの応答を表示
            st.write("Bot's response:")
            st.write(response)

# Settingsが選択された場合のレイアウト
elif selected == "Settings":
    st.title("Settings")
    st.write("Here you can configure the app settings.")
    # 必要に応じて設定項目を追加可能

