#libraries
import streamlit as st 
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain.llms import OpenAI

#session-states
if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []
if "input" not in st.session_state:
    st.session_state["input"] = []
if "stored_session" not in st.session_state:
    st.session_state["stored_session"] = []

def get_text():
    """
    Get the user input text.
    Returns:
        (str):The text entered by the user
    """
    input_text=st.text_input("You: ", st.session_state["input"], key="input",
                             placeholder="Your AI assistant here! Ask me anything ...",
                             label_visibility='hidden')
    return input_text



st.title("Mmemory Bot")

api=st.sidebar.text_input("API-Key", type="password")
MODEL=st.sidebar.selectbox(label='Model',options=['gpt-3.5-turbo','text-davinci-003','text-davinci-002'])
if api:

    llm=OpenAI(
        temperature=0,
        open_ai_key=api,
        model_name= MODEL,
    )

    if "entity_memory" not in st.session_state:
        st.session_state.entity_memory = ConversationEntityMemory(llm=llm,k=10)
    
    Conversation = ConversationChain(
        llm = llm,
        prompt = ENTITY_MEMORY_CONVERSATION_TEMPLATE,
        memory = st.session_state.entity_memory
    )
else:
    st.error("No API found")

user_input=get_text()

if user_input:
    output=Conversation.run(input=user_input)

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

with st.expander("Conversation"):
    for i in range(len(st.session_state['generated'])-1,-1,-1):
        st.info(st.session_state["past"][i]")
        st.info(st.session_state["generated"][i]")
