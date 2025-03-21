import os
import asyncio
import numpy as np
import chromadb
import autogen
import uvicorn
import pdfplumber
import tiktoken
import google.generativeai as genai
from fastapi import FastAPI, WebSocket
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# ✅ Configure Google Gemini API
genai.configure(api_key="AIzaSyCQVigwJJUkCotk2l_xQcEKe6MisptV4FI")
model = genai.GenerativeModel("gemini-1.5-flash")

config_list = [
    {
        "model": "gemini-1.5-flash",
        "api_key": "AIzaSyCQVigwJJUkCotk2l_xQcEKe6MisptV4FI",
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "price": [0.0001, 0.0002]
    }
]

# ✅ Ensure ChromaDB index directory exists
INDEX_DIR = "./chroma_index"
os.makedirs(INDEX_DIR, exist_ok=True)

# ✅ Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path=INDEX_DIR)
collection = chroma_client.get_or_create_collection(name="pdf_chunks")

# ✅ Autogen AI Agents
summarizer_agent = autogen.AssistantAgent(
    name="Summarizer",
    system_message="You are responsible for summarizing the extracted text from the PDF.",
    llm_config={"config_list": config_list}
)

user_proxy = autogen.UserProxyAgent(
    name="User_Proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=1,
    code_execution_config={"work_dir": ".", "use_docker": False},
)

# ✅ FastAPI App
app = FastAPI()


# ✅ Function to Extract Text from PDF
def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    text = ""
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"🚨 Error reading PDF: {str(e)}")
        return f"Error: Unable to extract text from the PDF due to {str(e)}"

    if not text.strip():
        return "Error: No readable text found in the PDF. It might be an image-based PDF."

    return text


# ✅ Function to Chunk Text
def chunk_text(text, max_tokens=512):
    """Splits text into chunks based on token limits."""
    tokenizer = tiktoken.get_encoding("cl100k_base")
    words = text.split()
    chunks, current_chunk = [], []
    current_length = 0

    for word in words:
        token_length = len(tokenizer.encode(word))
        if current_length + token_length > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(word)
        current_length += token_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# ✅ Load Sentence Transformer Model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# ✅ Function to Generate Embeddings
def get_embeddings(chunks):
    """Generates embeddings for each chunk."""
    return embedding_model.encode(chunks, convert_to_numpy=True)


# ✅ Function to Retrieve Relevant Chunks using Cosine Similarity
def retrieve_relevant_chunks(query, chunks, embeddings, top_k=3):
    """Finds the most relevant text chunks based on cosine similarity."""
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    similarities = np.dot(embeddings, query_embedding.T).flatten()
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [chunks[i] for i in top_indices]


# ✅ Initialize ChromaDB Client & Collection
chroma_client = chromadb.PersistentClient(path="./chroma_db")
embedding_function = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = chroma_client.get_or_create_collection(name="pdf_summaries", embedding_function=embedding_function)


# ✅ Function to Add Chunks to ChromaDB
def add_to_chroma(chunks):
    """Stores text chunks in ChromaDB."""
    for i, chunk in enumerate(chunks):
        collection.add(
            ids=[str(len(collection.get()["ids"]) + i)],
            documents=[chunk]
        )


# ✅ Function to Retrieve Similar Chunks from ChromaDB
def retrieve_from_chroma(query, top_k=3):
    """Retrieves similar text chunks from ChromaDB."""
    results = collection.query(query_texts=[query], n_results=top_k)
    return results["documents"][0] if results and "documents" in results else []


# ✅ Asynchronous Function to Summarize PDF

async def summarize_pdf_with_feedback(pdf_path, websocket=None):
    """Extracts text, chunks it, retrieves relevant parts, and summarizes, while keeping the feedback flow."""
    summary = None
    text = extract_text_from_pdf(pdf_path)
    
    if "Error" in text:
        print("🚨 Text extraction failed! Not calling the API.")
        await websocket.send_text("❌ Error extracting text from PDF. It may be an image-based PDF or corrupted.")
        return
    
    # 🔹 New: Chunking and Embedding
    chunks = chunk_text(text)
    embeddings = get_embeddings(chunks)

    # 🔹 New: Retrieve most relevant chunks
    relevant_chunks = retrieve_relevant_chunks("Summarize the document", chunks, embeddings)
    user_messages = "\n".join(relevant_chunks)  # Using relevant chunks for summary

    print(f"\n🔍 Sending Relevant Chunks to Summarizer:\n{user_messages[:500]}...\n")  # Debugging

    # Step 2: Generate Summary
    response = None
    try:
        response = await user_proxy.a_initiate_chat(
            recipient=summarizer_agent,
            message=user_messages
        )

        if hasattr(response, "chat_history") and isinstance(response.chat_history, list):
            chat_history = response.chat_history
            if len(chat_history) > 0:
                last_message = chat_history[-1]
                if isinstance(last_message, dict) and "content" in last_message:
                    summary = last_message["content"].strip()
                else:
                    summary = "❌ No valid content found in chat history."
            else:
                summary = "❌ Empty chat history received."

        elif isinstance(response, str) and response.strip():
            summary = response.strip()

        elif isinstance(response, dict) and "content" in response:
            summary = response["content"].strip()

        elif isinstance(response, list) and len(response) > 0:
            for item in response:
                if isinstance(item, dict) and "content" in item:
                    summary = item["content"].strip()
                    break
            else:
                summary = "❌ No valid content found in response list."

        else:
            summary = "❌ No valid summary format received."

        print(f"\n✅ Debug: Extracted Summary Before Sending:\n{summary}\n")

    except Exception as e:
        print(f"🚨 Error: LLM response failed due to {str(e)}")
        summary = "❌ Error: Unable to generate a summary due to an issue with the LLM response."

    # ✅ Feedback System Remains Unchanged
    if websocket and summary and summary.strip() and "Error" not in summary:
        print(f"\n📄 Sending Summary via WebSocket:\n{summary}\n")
        await websocket.send_text(f"\n📄 Summary:\n{summary}")
    else:
        print(f"\n❌ Summary could not be generated. Sending error message.\n")
        await websocket.send_text("❌ Error: Unable to generate a summary. Please try again later.")

    # ✅ Keeping User Feedback System as is
    await websocket.send_text("🤖 Are you satisfied with the summary? Reply with 'YES' or 'NO'.")

    feedback = await websocket.receive_text()
    if feedback.lower() == "no":
        await websocket.send_text("🔄 Please specify what changes you'd like.")
        user_input = await websocket.receive_text()
        await websocket.send_text(f"📌 Noted: {user_input}. We'll refine the summary accordingly!")
        refined_summary = await refine_summary(summary, user_input)
        await websocket.send_text(f"\n📄 Refined Summary:\n{refined_summary}")

    return summary


# ✅ WebSocket Endpoint for Real-Time PDF Processing
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint to receive PDF path and return the summary."""
    await websocket.accept()
    print("✅ WebSocket Connection Established!")

    try:
        while True:
            pdf_path = await websocket.receive_text()
            pdf_path = pdf_path.strip()
            print(f"📂 Received PDF path: {pdf_path}")

            await websocket.send_text(f"Processing PDF: {pdf_path}...")

            summary = await summarize_pdf_with_feedback(pdf_path, websocket)

            await websocket.send_text(f"\n📄 Summary:\n{summary}")
    except Exception as e:
        print(f"🚨 WebSocket Error: {e}")
        await websocket.send_text(f"Error: {e}")


# ✅ Asynchronous Function to Refine Summary
async def refine_summary(original_summary, user_feedback):
    """Refines the summary based on user feedback using the LLM model."""

    user_messages = f"Refine the following summary based on this feedback:\n\nFeedback: {user_feedback}\n\nOriginal Summary:\n{original_summary}"

    try:
        response = await user_proxy.a_initiate_chat(
            recipient=summarizer_agent,
            message=user_messages
        )

        refined_summary = None

        if hasattr(response, "chat_history") and isinstance(response.chat_history, list):
            chat_history = response.chat_history
            if len(chat_history) > 0:
                last_message = chat_history[-1]  # Get the last message
                if isinstance(last_message, dict) and "content" in last_message:
                    refined_summary = last_message["content"].strip()
                else:
                    refined_summary = "❌ No valid content found in chat history."
            else:
                refined_summary = "❌ Empty chat history received."

        elif isinstance(response, str) and response.strip():
            refined_summary = response.strip()

        elif isinstance(response, dict) and "content" in response:
            refined_summary = response["content"].strip()

        elif isinstance(response, list) and len(response) > 0:
            for item in response:
                if isinstance(item, dict) and "content" in item:
                    refined_summary = item["content"].strip()
                    break
            else:
                refined_summary = "❌ No valid content found in response list."

        else:
            refined_summary = "❌ Error: Unable to refine summary. No valid response format received."

    except Exception as e:
        print(f"🚨 Error refining summary: {str(e)}")
        refined_summary = f"❌ Error: Unable to refine summary due to {str(e)}"

    print(f"\n🔄 Refined Summary:\n{refined_summary}\n")  # Debugging
    return refined_summary

# ✅ Function to Update ChromaDB with Refined Summary
def update_chroma_summary(old_summary, new_summary):
    """Updates the vector database with refined summary."""
    collection.delete(where={"documents": old_summary})
    collection.add(
        ids=[str(len(collection.get()["ids"]))],
        documents=[new_summary]
    )


# ✅ Root Endpoint
@app.get("/")
async def read_root():
    return {"message": "FastAPI WebSocket PDF Summarizer"}


# ✅ Start FastAPI Server
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
