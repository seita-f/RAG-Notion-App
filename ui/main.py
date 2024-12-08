"""
UI using streamlit
"""
import streamlit as st
from streamlit_option_menu import option_menu  # サイドメニューのライブラリ

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
    
    # 会話エリア
    chat_container = st.container()
    with chat_container:
        # これまでの会話履歴
        if "conversation" in st.session_state and st.session_state.conversation:
            for message in st.session_state.conversation:
                if message["role"] == "user":
                    st.text(f"User: {message['content']}")
                else:
                    st.text(f"Bot: {message['content']}")

        # 入力フォーム
        user_input = st.text_input("Enter your message:", key="user_input")
        if st.button("Send"):
            if user_input:
                # 入力を会話メモリに保存
                st.session_state.memory.add_user_message(user_input)

                # 簡易応答生成（ここをRAG対応に変更可能）
                bot_response = f"Echo: {user_input}"
                st.session_state.memory.add_ai_message(bot_response)

                # 会話履歴を保存
                if "conversation" not in st.session_state:
                    st.session_state.conversation = []
                st.session_state.conversation.append({"role": "user", "content": user_input})
                st.session_state.conversation.append({"role": "bot", "content": bot_response})

                # 会話内容を更新して表示
                st.experimental_rerun()

# Settingsが選択された場合のレイアウト
elif selected == "Settings":
    st.title("Settings")
    st.write("Here you can configure the app settings.")
    # 必要に応じて設定項目を追加可能

