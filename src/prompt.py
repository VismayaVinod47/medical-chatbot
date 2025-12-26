from langchain_core.prompts import ChatPromptTemplate

system_prompt = (
    "You are a medical information assistant. "
    "Use the provided context to answer the question. "
    "If you do not know the answer, say you do not know. "
    "Use three sentences maximum and keep the answer concise.\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{question}")   # ✅ MUST be question
    ]
)
