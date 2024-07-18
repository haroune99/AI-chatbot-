import os
import PyPDF2
import openai
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import getpass

# Set OpenAI API Key
os.environ["OPENAI_API_KEY"] = getpass.getpass()

# Define the folder path where your PDF documents are stored
folder_path = "/Users/harouneaaffoute/Documents/OpenAI/summarized_tickets"

# Initialize an empty list to store document contents
documents = []

# Iterate through each file in the folder
for filename in os.listdir(folder_path):
    # Ensure the file is a PDF file (you can add more file format checks if needed)
    if filename.endswith(".pdf"):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "rb") as file:
            # Create a PDF reader object
            pdf_reader = PyPDF2.PdfReader(file)
            # Extract text from each page using list comprehension
            pages_text = [page.extract_text() for page in pdf_reader.pages]
            # Join the text from all pages into a single string
            pdf_text = "\n".join(pages_text)
            # Append the extracted text to the list
            documents.append(pdf_text)

text_splitter = CharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separator="\n",
)

chunks = []

for text in documents:
    split_text = text_splitter.split_text(text)
    chunks.append(split_text)

# Using list comprehension
chunks = [item for sublist in chunks for item in sublist]

class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata

metadata = {"title": "text", "author": "Client central"}

# Initialize an empty list to store Document objects
chunks_documents = []

# Iterate through each chunk and create a Document object for each one
for chunk in chunks:
    document = Document(chunk, metadata)
    chunks_documents.append(document)

vectorstore = Chroma.from_documents(chunks_documents, embedding=OpenAIEmbeddings(), persist_directory="my_embed_summzrized_tickets")
vectorstore.persist()
retriever = vectorstore.as_retriever()

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

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
qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question.

{context}"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


### Statefully manage chat history ###
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# Load cross-encoder model for re-ranking
cross_encoder_model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/ms-marco-MiniLM-L-6-v2')
cross_encoder_tokenizer = AutoTokenizer.from_pretrained('cross-encoder/ms-marco-MiniLM-L-6-v2')

def re_rank_documents(query, documents):
    scores = []
    for doc in documents:
        inputs = cross_encoder_tokenizer.encode_plus(query, doc.page_content, return_tensors='pt', max_length=512, truncation=True)
        outputs = cross_encoder_model(**inputs)
        scores.append(outputs.logits.item())
    scored_documents = list(zip(documents, scores))
    scored_documents = sorted(scored_documents, key=lambda x: x[1], reverse=True)
    re_ranked_documents = [doc for doc, score in scored_documents]
    return re_ranked_documents

def get_re_ranked_documents(query):
    # Replace with your actual retrieval logic using vectorstore or retriever
    retrieved_docs = retriever.get_relevant_documents(query)
    re_ranked_docs = re_rank_documents(query, retrieved_docs)
    return re_ranked_docs
