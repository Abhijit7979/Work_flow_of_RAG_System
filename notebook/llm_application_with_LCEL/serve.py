

from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import os
from langserve import add_routes
from dotenv import load_dotenv
# Import BaseModel from Pydantic
from pydantic import BaseModel
import uvicorn

load_dotenv()

groq_api_key=os.getenv("GROQ_API_KEY")
model=ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

# 1. Create prompt template
system_template = "Translate the following into {language}:"
prompt_template = ChatPromptTemplate.from_messages([
    ('system', system_template),
    ('user', '{text}')
])

parser=StrOutputParser()

# Create chain
chain=prompt_template|model|parser

# 2. ✅ Define Pydantic Input and Output models
class Input(BaseModel):
    language: str
    text: str

class Output(BaseModel):
    output: str # The default output key for a single-output chain is "output"

# 3. ✅ Add types to your chain
chain = chain.with_types(input_type=Input, output_type=Output)

# App definition
app=FastAPI(
    title="Langchain Server",
    version="1.0",
    description="A simple API server using Langchain runnable interfaces"
)

# Adding chain routes
add_routes(
    app,
    chain, # Pass the typed chain
    path="/chain",
    enabled_endpoints=["invoke", "stream", "playground","input_schema",'output_schema','config_schema','stream_log','stream_events']
)

if __name__=="__main__":
    uvicorn.run(app,host="127.0.0.1",port=8017)