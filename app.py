import streamlit as st
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain_community.chat_models import ChatOpenAI

st.set_page_config(page_title="OpenRouter Chatbot", layout="centered")
st.title("🧠 Chatbot with Context Memory")

# Initialize session state
for key in ["generated", "past"]:
    if key not in st.session_state:
        st.session_state[key] = []

api = st.sidebar.text_input("🔐 OpenRouter API Key", type="password")
MODEL = st.sidebar.selectbox("Choose a Model", [
    "meta-llama/llama-3-8b-instruct",
    "anthropic/claude-3-haiku",
    "google/gemini-pro"
])

if api:
    try:
        llm = ChatOpenAI(
            model_name=MODEL,
            temperature=0,
            openai_api_key=api,
            openai_api_base="https://openrouter.ai/api/v1"
        )

        if "entity_memory" not in st.session_state:
            st.session_state.entity_memory = ConversationEntityMemory(llm=llm, k=10)

        if "conversation" not in st.session_state:
            st.session_state.conversation = ConversationChain(
                llm=llm,
                prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
                memory=st.session_state.entity_memory
            )

        if st.sidebar.button("🧹 Clear Conversation"):
            st.session_state.past = []
            st.session_state.generated = []
            st.session_state.entity_memory = ConversationEntityMemory(llm=llm, k=10)
            st.session_state.conversation = ConversationChain(
                llm=llm,
                prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
                memory=st.session_state.entity_memory
            )
            st.success("Conversation memory cleared.")

        # Display previous messages
        for i in range(len(st.session_state.generated)):
            with st.chat_message("user"):
                st.markdown(st.session_state.past[i])
            with st.chat_message("assistant"):
                st.markdown(st.session_state.generated[i])

        # Chat input (auto-clears!)
        user_input = st.chat_input("Ask something...")
        if user_input:
            with st.chat_message("user"):
                st.markdown(user_input)
            output = st.session_state.conversation.run(user_input)
            with st.chat_message("assistant"):
                st.markdown(output)

            st.session_state.past.append(user_input)
            st.session_state.generated.append(output)

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.error("🔑 Please enter your OpenRouter API key to begin.")

