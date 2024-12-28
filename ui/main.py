import sys
import os
import streamlit as st
from streamlit_option_menu import option_menu

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llm import RAGApp

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User Interface
st.set_page_config(page_title="RAG Desktop App", layout="wide")
st.markdown(
    """
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css" rel="stylesheet">
    """,
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    selected = option_menu(
        menu_title="Menu",
        options=["Chat", "Settings"],
        icons=["chat", "gear"],  # Font Awesome
        menu_icon="menu-down",
        default_index=0,
    )

# Initialize session state for temperature
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.3  # default temperature

# Chat Layout
if selected == "Chat":

    st.title("Chat with RAG Bot")

    # First loading message
    if not st.session_state.chat_history:
        st.session_state.chat_history.append({"user": None, "bot": "How can I help you?"})

    # Display conversation history
    chat_container = st.container()
    with chat_container:
        for chat in st.session_state.chat_history:
            if chat["user"]:
                st.markdown(f"**You:** {chat['user']}")
            if chat["bot"]:
                st.markdown(
                    f"""
                    <div style='display: flex; align-items: flex-start; margin-bottom: 10px;'>
                        <div style='font-size: 24px; margin-right: 10px;'>
                            <i class="fa fa-robot" style="color: #555;"></i>  <!-- Robot-icon -->
                        </div>
                        <div style='background-color: #f0f0f0; padding: 10px; border-radius: 10px;'>
                            {chat["bot"]}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    # User Input
    user_input = st.text_area(
        label="Placeholder Label", 
        placeholder="Type your question here...", 
        height=100, 
        label_visibility="collapsed"
    )

    send_button = st.button("Send", key="send_button")

    if send_button and user_input.strip():
        # Spinner
        with st.spinner("Generating response..."):

            # DEBUG:
            print(f"##### Passing temp to RAG: {st.session_state.temperature} #####")

            # RAGApp
            rag_app = RAGApp(temperature=st.session_state.temperature)
            response = rag_app.get_response(user_input)

        # Update chat history
        st.session_state.chat_history.append({"user": user_input, "bot": response})

        # Dispaly the updated convo
        with chat_container:
            st.markdown(f"**You:** {user_input}")
            st.markdown(
                f"""
                <div style='display: flex; align-items: flex-start; margin-bottom: 10px;'>
                    <div style='font-size: 24px; margin-right: 10px;'>
                            <i class="fa fa-robot" style="color: #555;"></i>  <!-- Robot-icon -->
                    </div>
                    <div style='background-color: #f0f0f0; padding: 10px; border-radius: 10px;'>
                        {response}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

# Setting layout
elif selected == "Settings":
    st.title("Settings")
    st.write("Here you can configure the app settings.")
    st.write("#### Adjust the creativity of responses:")
    # Temperature slider
    st.session_state.temperature = st.slider(
        "Temperature (0.1-1.0):",   # label
        min_value=0.1,              # min
        max_value=1.0,              # max
        value=st.session_state.temperature,  # default temperature
        step=0.1                    # step
    )
