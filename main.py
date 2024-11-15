import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders.text import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict
from langchain_postgres import PGVector
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector

llm = ChatGroq(model="llama3-8b-8192", temperature=0)

load_dotenv()
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"] 

text_list = []

transcript = YouTubeTranscriptApi.get_transcript('7TWKKww-F30', languages=['en', 'ar', 'as', 'bg', 'zh-Hans', 'zh-Hant'])

for _ in transcript:
    text = (_['text'])
    text_list.append(text)

    with open("transcript.txt", "a") as opf:
        opf.write(text)

loader = TextLoader(file_path="./transcript.txt")

docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
embedding.embed_query("Give a summary of the retrieved data")

# See docker command above to launch a postgres instance with pgvector enabled.
connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"  # Uses psycopg3!
collection_name = "transcript.txt"

vectorstore = PGVector(
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
    collection_name=collection_name,
    connection=connection,
    use_jsonb=True,
)

vectorstore.add_documents(documents=splits, embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001"))

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever(search_type = "similarity", search_kwargs={"k": 6})

### Contextualize question ###
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)


### Answer question ###
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


### Statefully manage chat history ###
class State(TypedDict):
    input: str
    chat_history: Annotated[list, add_messages]
    context: str
    answer: str


def call_model(state: State):
    response = rag_chain.invoke(state)
    return {
        "chat_history": [
            HumanMessage(state["input"]),
            AIMessage(response["answer"]),
        ],
        "context": response["context"],
        "answer": response["answer"],
    }


workflow = StateGraph(state_schema=State)
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "abc123"}}


print("Chatbot initialized. Type 'exit' to stop.")
while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit']:
        print("Exiting...")
        break

    result = app.invoke(
        {"input": user_input},
        config=config,
    )
    print(result["answer"])

