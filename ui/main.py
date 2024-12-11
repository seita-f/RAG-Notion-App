"""
UI using streamlit
"""
import sys
import os
import streamlit as st
from streamlit_option_menu import option_menu  

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llm import RAGApp

# ユーザインターフェース
st.set_page_config(page_title="RAG Desktop App", layout="wide")

# サイドバー
with st.sidebar:
    selected = option_menu(
        menu_title="Menu",
        options=["Chat", "Settings"],
        icons=["chat", "gear"],  # Font Awesome
        menu_icon="menu-down",
        default_index=0,
    )

# Initialize session state
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.3  # default temperature

# チャットレイアウト
if selected == "Chat":
    st.title("Chat with RAG Bot")
    
    # ユーザ入力
    st.markdown("### Enter your message:")
    user_input = st.text_area(
        label="",
        placeholder="Type your question here...",
        height=100,  
    )

    if st.button("Send"):
        if user_input.strip():
            # スピナー
            with st.spinner("Generating response..."):

                # DEBUG:
                print(f"##### Passing temp to RAG: {st.session_state.temperature} #####")

                # RAGApp
                rag_app = RAGApp(temperature=st.session_state.temperature)
                response = rag_app.get_response(user_input)

            st.markdown("### Bot's response:")
            st.markdown(response, unsafe_allow_html=True)  

# 設定レイアウト
elif selected == "Settings":
    st.title("Settings")
    st.write("Here you can configure the app settings.")
    st.write("#### Adjust the creativity of responses:")
    # Temperature スライダー
    st.session_state.temperature = st.slider(
        "Temperature (1-10):",  # ラベル
        min_value=0.1,          # 最小値
        max_value=1.0,          # 最大値
        value=st.session_state.temperature,  # デフォルト temperature
        step=0.1                # ステップ
    )
