"""
UI using streamlit
"""
import sys
import os
import streamlit as st
from streamlit_option_menu import option_menu  

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llm import RAGApp

# User Interface
st.set_page_config(page_title="RAG Desktop App", layout="wide")

# Side bar
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

# Layout for Chat
if selected == "Chat":
    st.title("Chat with RAG Bot")
    
    # Input
    st.markdown("### Enter your message:")
    
    # Text area
    user_input = st.text_area(
        label="",
        placeholder="Type your question here...",
        height=100,  
    )

    if st.button("Send"):
        if user_input.strip():
            # spinner
            with st.spinner("Generating response..."):
                # RAGApp 
                rag_app = RAGApp(config_path="config.yml", temperature=st.session_state.temperature)
                response = rag_app.get_response(user_input)

            # Display result
            st.markdown("### Bot's response:")
            st.markdown(response, unsafe_allow_html=True)  # Enable HTML rendering

# Layout for Settings
elif selected == "Settings":
    st.title("Settings")
    st.write("Here you can configure the app settings.")
    st.write("#### Adjust the creativity of responses:")
    # Temperature slider
    st.session_state.temperature = st.slider(
        "Temperature (1-10):",  # label
        min_value=0.1,          # min 
        max_value=1.0,          # max
        value=st.session_state.temperature,     # default
        step=0.1                # step
    )
