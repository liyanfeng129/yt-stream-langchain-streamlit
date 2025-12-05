import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from load_vectors import VectorStoreLoader
from reranker_retriever import create_reranked_retriever
from langchain_core.runnables import RunnableLambda


load_dotenv()

# app config
st.set_page_config(page_title="Streaming bot", page_icon="🤖")
st.title("Streaming bot")

@st.cache_resource
def load_vectorstore_and_retriever():
    """Load and cache the vectorstore and retriever."""
    vector_store_loader = VectorStoreLoader()
    vector_store_loader.load()
    vector_store = vector_store_loader.vectorstore
    reranked_retriever = create_reranked_retriever(vector_store=vector_store)
    return vector_store, reranked_retriever

vector_store, reranked_retriever = load_vectorstore_and_retriever()

def get_response(user_query, chat_history):

    template = """
    You are a helpful assistant. Answer the following questions considering the history of the conversation and the provided context:

    Chat history: {chat_history}

    Query context: {query_context}

    User question: {user_question}

    answer in english.

    """

    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatOllama(
    model="qwen3:8b",
    temperature=0.6,
    )

    def retrieve_context(inputs):
        docs = reranked_retriever.invoke({"user_question": inputs["user_question"]})
        inputs["query_context"] = docs
        return inputs
    
    context_retrieval = RunnableLambda(retrieve_context)
       
    chain = context_retrieval | prompt | llm | StrOutputParser()
    
    return chain.stream({
        "chat_history": chat_history,
        "user_question": user_query,
    })

# session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am a bot. How can I help you?"),
    ]

    
# conversation
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

# user input
user_query = st.chat_input("Type your message here...")
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        response = st.write_stream(get_response(user_query, st.session_state.chat_history))

    st.session_state.chat_history.append(AIMessage(content=response))
