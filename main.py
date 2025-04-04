from fastapi import FastAPI, UploadFile, File, HTTPException, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import uuid
import shutil
import datetime
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
    UnstructuredExcelLoader,
    WebBaseLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import json

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create uploads directory if it doesn't exist
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Create vector stores directory if it doesn't exist
VECTORSTORE_DIR = "vectorstores"
os.makedirs(VECTORSTORE_DIR, exist_ok=True)

# Create URLs directory if it doesn't exist
URL_DIR = "urls"
os.makedirs(URL_DIR, exist_ok=True)

# Store chat sessions
chat_sessions = {}

# Initialize OpenAI embeddings and model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", api_key=OPENAI_API_KEY)


class ChatRequest(BaseModel):
    # message: str
    history: Optional[List[dict]] = []
    message: str
    session_id: Optional[str] = None


class DocInfo(BaseModel):
    id: str
    filename: str
    type: str


class UrlRequest(BaseModel):
    url: str


@app.get("/")
def read_root():
    return {"message": "Document Chat API is running"}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    # Generate a unique ID for the document
    doc_id = str(uuid.uuid4())

    # Determine file type
    filename = file.filename
    file_extension = os.path.splitext(filename)[1].lower()

    # Create directory for this document
    doc_dir = os.path.join(UPLOAD_DIR, doc_id)
    os.makedirs(doc_dir, exist_ok=True)

    # Save the file
    file_path = os.path.join(doc_dir, filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Get file size
    file_size = os.path.getsize(file_path)

    # Process file in the background
    if background_tasks:
        background_tasks.add_task(process_document, doc_id, file_path, file_extension)
    else:
        # For synchronous processing (useful for testing)
        await process_document(doc_id, file_path, file_extension)

    # Save metadata
    metadata = {
        "id": doc_id,
        "name": filename,
        "type": file_extension[1:],  # Remove the dot from extension
        "uploadedAt": datetime.datetime.now().isoformat(),
        "size": file_size,
        "source": "file"
    }

    with open(os.path.join(doc_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f)

    return metadata


@app.post("/add_url")
async def add_url(url_data: UrlRequest, background_tasks: BackgroundTasks = None):
    # Generate a unique ID for the URL
    url_id = str(uuid.uuid4())

    # Extract the URL
    url = url_data.url

    # Create directory for this URL
    url_dir = os.path.join(URL_DIR, url_id)
    os.makedirs(url_dir, exist_ok=True)

    # Save URL to a file
    url_file_path = os.path.join(url_dir, "url.txt")
    with open(url_file_path, "w") as f:
        f.write(url)

    # Extract domain for name
    try:
        from urllib.parse import urlparse
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        path = parsed_url.path
        name = domain + (path if path and path != "/" else "")
    except:
        name = url[:50]  # Use part of URL as name if parsing fails

    # Create metadata
    metadata = {
        "id": url_id,
        "name": name,
        "type": "url/html",
        "uploadedAt": datetime.datetime.now().isoformat(),
        "size": len(url),  # Use URL length as initial size
        "source": "url",
        "sourceUrl": url
    }

    # Save metadata
    with open(os.path.join(url_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f)

    # Process URL in the background
    if background_tasks:
        background_tasks.add_task(process_url, url_id, url)
    else:
        # For synchronous processing (useful for testing)
        await process_url(url_id, url)

    return metadata


@app.get("/documents")
async def get_documents():
    docs = []

    # Get uploaded documents
    if os.path.exists(UPLOAD_DIR):
        for doc_id in os.listdir(UPLOAD_DIR):
            doc_dir = os.path.join(UPLOAD_DIR, doc_id)
            if os.path.isdir(doc_dir):
                metadata_path = os.path.join(doc_dir, "metadata.json")
                if os.path.exists(metadata_path):
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                        docs.append(metadata)
                else:
                    # Fallback for documents without metadata
                    files = os.listdir(doc_dir)
                    if files:
                        filename = next((f for f in files if f != "metadata.json"), None)
                        if filename:
                            file_path = os.path.join(doc_dir, filename)
                            file_extension = os.path.splitext(filename)[1].lower()[1:]
                            docs.append({
                                "id": doc_id,
                                "name": filename,
                                "type": file_extension,
                                "size": os.path.getsize(file_path),
                                "uploadedAt": datetime.datetime.fromtimestamp(os.path.getctime(file_path)).isoformat(),
                                "source": "file"
                            })

    # Get URLs
    if os.path.exists(URL_DIR):
        for url_id in os.listdir(URL_DIR):
            url_dir = os.path.join(URL_DIR, url_id)
            if os.path.isdir(url_dir):
                metadata_path = os.path.join(url_dir, "metadata.json")
                if os.path.exists(metadata_path):
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                        docs.append(metadata)

    return docs


@app.post("/chat")
async def chat(request: ChatRequest):

    # Get the message from the messages array
    if not request.message:
        raise HTTPException(status_code=400, detail="No message provided")

    message = request.message
    session_id = request.session_id

    # Check if session exists
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Chat session not found")

    # Get the retrieval chain for this session
    retrieval_chain = chat_sessions[session_id]

    # Convert history to the format expected by LangChain
    formatted_history = []
    for entry in request.history:
        if entry.get("role") == "user":
            formatted_history.append((entry.get("content"), ""))
        elif entry.get("role") == "assistant":
            # Adjust the last entry to include the assistant's response
            if formatted_history:
                last_user_msg, _ = formatted_history.pop()
                formatted_history.append((last_user_msg, entry.get("content")))
    print(f"Formatted history: {formatted_history}")
    try:
        # Generate response
        response = retrieval_chain.invoke({
            "question": message,
            "chat_history": formatted_history
        })

        return {
            "role": "assistant",
            "content": response["answer"],
            "source_documents": response.get("source_documents", [])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/create_session/{doc_id}")
async def create_session(doc_id: str):
    # Check if document exists in uploads directory
    doc_dir = os.path.join(UPLOAD_DIR, doc_id)
    url_dir = os.path.join(URL_DIR, doc_id)
    print(f"Document path: {doc_dir}")
    print(f"URL path: {url_dir}")

    is_document = os.path.exists(doc_dir)
    is_url = os.path.exists(url_dir)

    if not (is_document or is_url):
        raise HTTPException(status_code=404, detail="Document or URL not found")

    # Determine the correct vectorstore path
    if is_document:
        vectorstore_path = os.path.join(VECTORSTORE_DIR, doc_id)
    else:
        vectorstore_path = os.path.join(VECTORSTORE_DIR, f"url_{doc_id}")

    print(f"Vectorstore path: {vectorstore_path}")

    if not os.path.exists(vectorstore_path):
        raise HTTPException(status_code=404, detail="Content not processed yet")

    try:
        # Load the vector store
        vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
        print("Vectorstore loaded successfully")

        # Create a conversation memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        print("Memory created successfully")

        # Create a retrieval chain
        retrieval_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory
        )
        print("Retrieval chain created successfully")

        # Store the retrieval chain
        chat_sessions[doc_id] = retrieval_chain

        return {"id": doc_id, "status": "created"}
    except Exception as e:
        print(f"Error creating session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    # Check if it's a document
    doc_dir = os.path.join(UPLOAD_DIR, doc_id)
    if os.path.exists(doc_dir):
        # Remove document directory
        shutil.rmtree(doc_dir, ignore_errors=True)

        # Remove vectorstore if exists
        vectorstore_path = os.path.join(VECTORSTORE_DIR, doc_id)
        if os.path.exists(vectorstore_path):
            shutil.rmtree(vectorstore_path, ignore_errors=True)

        # Remove chat session if exists
        if doc_id in chat_sessions:
            del chat_sessions[doc_id]

        return {"status": "deleted"}

    # Check if it's a URL
    url_dir = os.path.join(URL_DIR, doc_id)
    if os.path.exists(url_dir):
        # Remove URL directory
        shutil.rmtree(url_dir, ignore_errors=True)

        # Remove vectorstore if exists
        vectorstore_path = os.path.join(VECTORSTORE_DIR, f"url_{doc_id}")
        if os.path.exists(vectorstore_path):
            shutil.rmtree(vectorstore_path, ignore_errors=True)

        # Remove chat session if exists
        if doc_id in chat_sessions:
            del chat_sessions[doc_id]

        return {"status": "deleted"}

    # If neither document nor URL exists
    raise HTTPException(status_code=404, detail="Document or URL not found")


async def process_document(doc_id: str, file_path: str, file_extension: str):
    try:
        # Load document based on file type
        if file_extension == ".pdf":
            loader = PyPDFLoader(file_path)
        elif file_extension in [".docx", ".doc"]:
            loader = Docx2txtLoader(file_path)
        elif file_extension == ".txt":
            loader = TextLoader(file_path)
        elif file_extension == ".csv":
            loader = CSVLoader(file_path)
        elif file_extension in [".xlsx", ".xls"]:
            loader = UnstructuredExcelLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

        # Load the document
        documents = loader.load()

        # Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)

        # Create a vector store
        vectorstore = FAISS.from_documents(chunks, embeddings)

        # Save the vector store
        vectorstore_path = os.path.join(VECTORSTORE_DIR, doc_id)
        vectorstore.save_local(vectorstore_path)

        # Update metadata with content size
        doc_dir = os.path.join(UPLOAD_DIR, doc_id)
        metadata_path = os.path.join(doc_dir, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            # Update with actual content size (sum of all chunks)
            content_size = sum(len(chunk.page_content) for chunk in chunks)
            metadata["size"] = content_size

            with open(metadata_path, "w") as f:
                json.dump(metadata, f)

        return True
    except Exception as e:
        print(f"Error processing document: {e}")
        return False


async def process_url(url_id: str, url: str):
    try:
        # Create a WebBaseLoader for the URL
        loader = WebBaseLoader(url)

        # Load the content
        documents = loader.load()

        # Get content size
        content_size = sum(len(doc.page_content) for doc in documents)

        # Update metadata with actual content size and title
        url_dir = os.path.join(URL_DIR, url_id)
        metadata_path = os.path.join(url_dir, "metadata.json")

        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            # Update size
            metadata["size"] = content_size

            # Try to extract title from first document
            if documents and hasattr(documents[0], 'metadata') and 'title' in documents[0].metadata:
                metadata["name"] = documents[0].metadata['title']

            with open(metadata_path, "w") as f:
                json.dump(metadata, f)

        # Split the content into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)

        # Create a vector store
        vectorstore = FAISS.from_documents(chunks, embeddings)

        # Save the vector store with a special prefix to distinguish from documents
        vectorstore_path = os.path.join(VECTORSTORE_DIR, f"url_{url_id}")
        vectorstore.save_local(vectorstore_path)

        return True
    except Exception as e:
        print(f"Error processing URL: {e}")
        return False


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
