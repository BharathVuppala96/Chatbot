import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory

# Initialize session state
if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you today?"}]

# Initialize model & chain
gemini_model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest",
    google_api_key=st.secrets["GOOGLE_API_KEY"]
)
conversation = ConversationChain(memory=st.session_state.buffer_memory, llm=gemini_model)

# UI
st.title("🗣️ Conversational Chatbot")
st.subheader("㈻ Simple Chat Interface for LLMs by Build Fast with AI")

# Chat input
if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state["last_prompt"] = prompt

# Display history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Generate assistant response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = conversation.predict(input=st.session_state["last_prompt"])
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

# Reset option
if st.button("Clear Chat"):
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you today?"}]
    st.session_state.buffer_memory.clear()
