import streamlit as st
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain_community.chat_models import ChatOpenAI

st.set_page_config(page_title="Memory Chatbot", layout="centered")
st.title("🧠 Memory Bot - Chat with Context")

for key in ["generated", "past"]:
    if key not in st.session_state:
        st.session_state[key] = []

def get_text():
    return st.text_input(
        "You:", 
        value="", 
        placeholder="Your AI assistant here! Ask me anything ...",
        label_visibility="hidden",
        key="input_text"
    )

api = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
MODEL = st.sidebar.selectbox("Choose a Model", ["gpt-3.5-turbo", "gpt-4"])

if api:
    try:
        llm = ChatOpenAI(
            temperature=0,
            openai_api_key=api,
            model_name=MODEL
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

        user_input = get_text()

        if user_input:
            output = st.session_state.conversation.run(input=user_input)
            st.session_state.past.append(user_input)
            st.session_state.generated.append(output)

        with st.expander("💬 Conversation History"):
            for i in range(len(st.session_state['generated']) - 1, -1, -1):
                st.markdown(f"**You:** {st.session_state['past'][i]}")
                st.markdown(f"**Bot:** {st.session_state['generated'][i]}")
    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.error("🔐 Please enter your OpenAI API key to begin.")
