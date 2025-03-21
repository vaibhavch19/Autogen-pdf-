
# 🧠 LLM-Powered PDF Summarizer with ChromaDB, Gemini, FastAPI & AutoGen

A powerful, modular PDF summarization system built using **Google Gemini LLM**, **ChromaDB vector search**, **SentenceTransformer embeddings**, **AutoGen agents**, and **FastAPI with WebSocket support**. This tool enables real-time semantic summarization, validation, and refinement of PDF documents using hybrid LLM + retrieval-based techniques.

---

## 🚀 Features

- 🔍 **Semantic Chunk Retrieval via ChromaDB**
- 🧠 **LLM-powered summarization using Gemini (AutoGen)**
- 📎 **PDF Text Extraction using pdfplumber**
- ✂️ **Chunking with Token Limit Awareness**
- 🔁 **Summary Refinement with User Feedback**
- 🔄 **Real-time WebSocket Interaction via FastAPI**
- 🧩 **Hybrid Search Support (Regex + Semantic Matching)**

---

## 📂 Project Structure

```
.
├── embeddings/
│   └── embedding_model.py     # SentenceTransformer logic
├── pdf_processing/
│   └── extractor.py           # PDF text extraction & chunking
├── vectorstore/
│   └── chroma_db.py           # ChromaDB collection creation & querying
├── llm_agents/
│   └── run_crew.py            # AutoGen Crew & Agents (Gemini)
├── api/
│   └── server.py              # FastAPI WebSocket Server
├── requirements.txt
└── README.md
```

---

## ⚙️ Technologies Used

| Component                | Tech Stack                                      |
|-------------------------|--------------------------------------------------|
| LLM                     | Google Gemini 1.5 Flash via AutoGen              |
| Embeddings              | SentenceTransformer (`all-MiniLM-L6-v2`)        |
| Vector Store            | ChromaDB                                         |
| PDF Text Extraction     | pdfplumber                                       |
| Agent Framework         | AutoGen (AssistantAgent, UserProxyAgent)        |
| Backend API             | FastAPI with WebSocket                          |

---

## 💡 How It Works

1. **PDF Uploaded ➝ Text Extracted (via pdfplumber)**
2. **Text ➝ Tokenized & Chunked (~512 tokens)**
3. **Chunks ➝ Embedded using SentenceTransformer**
4. **Embeddings ➝ Stored in ChromaDB**
5. **Query ➝ Embedding ➝ Top-k Similar Chunks Retrieved**
6. **LLM (Gemini) + Retrieved Chunks ➝ Summary Generated**
7. **User Feedback ➝ Summary Refined (via LLM)**

---

## 🛠️ Setup Instructions

1. **Clone the repo**
```bash
git clone https://github.com/yourusername/pdf-summarizer-llm.git
cd pdf-summarizer-llm
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set your Gemini API Key**
- Add your API key in `.env` or directly in `run_crew.py`:
```python
config_list = [
    {
        "model": "gemini/gemini-1.5-flash",
        "api_key": "YOUR_GEMINI_API_KEY"
    }
]
```

4. **Run the server**
```bash
python api/server.py
```

---

## 📡 WebSocket Workflow

- Client connects via WebSocket.
- Sends PDF and Query.
- Server processes PDF ➝ Embeds ➝ Retrieves ➝ Summarizes ➝ Sends Response.
- User provides feedback ➝ Server refines response.

---

## 🧪 Example Use Case

> _"Upload a contract PDF and ask for a summary of payment terms."_

- System identifies and retrieves relevant sections.
- Gemini summarizes key clauses.
- User can say “refine further focusing on due dates” ➝ Improved summary returned.

---

## 📌 Future Enhancements

- ✅ OCR support for scanned/image-based PDFs.
- ✅ Hybrid search: Combine Regex & Embeddings.
- ✅ Export summary as JSON/CSV.
- ✅ RAG pipelines using LangChain.

---

## 🙌 Acknowledgements

- [AutoGen by Microsoft](https://github.com/microsoft/autogen)
- [Google Gemini API](https://deepmind.google/discover/gemini/)
- [ChromaDB](https://www.trychroma.com/)
- [SentenceTransformers](https://www.sbert.net/)
- [FastAPI](https://fastapi.tiangolo.com/)

---

## 📄 License

This project is licensed under MIT. Feel free to fork, extend, and contribute.

---

## 💬 Feedback & Contributions

Pull requests, feedback, and feature suggestions are welcome! 🙌  
Feel free to create issues or contact me via [LinkedIn](https://linkedin.com/in/your-profile).
