from flask import Flask, render_template, jsonify, request
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

app=Flask(__name__)

load_dotenv()
PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
  
GEMINI_API_KEY=os.getenv("GEMINI_API_KEY")  

embeddings=download_embeddings()
index_name = "medical-chatbot"

docsearch=PineconeVectorStore.from_existing_index(
    
    embedding=embeddings,
    index_name=index_name
    )


retriever=docsearch.as_retriever(search_type="similarity",search_kwargs={"k":3})

from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",   # fast + free tier
    temperature=0.1
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{question}")   # ✅ MUST be question
    ]
)
from langchain_core.runnables import RunnablePassthrough

# define helper functions FIRST
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
qa_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
)
from flask import request

@app.route("/get", methods=["POST"])
def chat():
    user_question = request.form["msg"]
    print("User:", user_question)

    response = qa_chain.invoke(user_question)
    print("Bot:", response.content)

    # ✅ Return plain text (matches your JS)
    return response.content











os.environ["PINECONE_API_KEY"]=PINECONE_API_KEY
  
os.environ["GEMINI_API_KEY"]=GEMINI_API_KEY


@app.route("/")
def index():
    return render_template('chat.html')



if __name__=='__main__':
    app.run(host="0.0.0.0",port=5001,debug=True)



