from src.helper import load_pdf_files,filter_to_minimal_docs,text_split,download_embeddings

from dotenv import load_dotenv
import os
load_dotenv()



PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
# OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")  
GEMINI_API_KEY=os.getenv("GEMINI_API_KEY")  


os.environ["PINECONE_API_KEY"]=PINECONE_API_KEY
# os.environ["OPENAI_API_KEY"]=OPENAI_API_KEY     
os.environ["GEMINI_API_KEY"]=GEMINI_API_KEY
extracted_data=load_pdf_files(data='data/')
filter_data=filter_to_minimal_docs(extracted_data)
texts_chunk=text_split(filter_data)
embeddings=download_embeddings()


from pinecone import Pinecone
pinecone_api_key=PINECONE_API_KEY


pc=Pinecone(api_key=pinecone_api_key)



from pinecone import ServerlessSpec

index_name = "medical-chatbot"
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws",region="us-east-1")
        
    )
index=pc.Index(index_name)




from langchain_pinecone import PineconeVectorStore

docsearch=PineconeVectorStore.from_documents(
    documents=texts_chunk,
    embedding=embeddings,
    index_name=index_name
    )



#Load existing index
from langchain_pinecone import PineconeVectorStore  
docsearch=PineconeVectorStore.from_existing_index(
    embedding=embeddings,
    index_name=index_name
    )


