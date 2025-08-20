import os 
from dotenv import load_dotenv 
from langchain_community.llms import Ollama 
import streamlit as st 
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser

load_dotenv(dotenv_path="../.env")

# Langsmith Tracking 
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]='true'
os.environ["LANGCHAIN_PROJECT"] = "BASIC_GENAI_APP"


# Prompt Templates 
prompt_temp=ChatPromptTemplate.from_messages(
    [
        ("system","you are a helpful assistant.Please respond to the question asked."),
        ("user","Question:{user_question}")
    ]
)


# Streamlit framework

st.title("This is langchin chatbot using llama 3")
input_text=st.text_input("what question you have in mind ??")


## Ollama Llama3 model
llm=Ollama(model="llama3")

output_text=StrOutputParser()
chain=prompt_temp|llm|output_text

if input_text:
    st.write(chain.invoke({"user_question":input_text}))