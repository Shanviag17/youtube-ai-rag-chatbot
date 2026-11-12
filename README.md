# YouTube AI — Chat with a Video using RAG

An experimental AI project exploring **Retrieval-Augmented Generation (RAG)** through YouTube video transcripts.

The goal is simple: instead of watching an entire video to find a specific answer, users can ask questions about its content and receive context-aware responses supported by relevant transcript excerpts and timestamps.

This project focuses primarily on understanding and implementing the RAG pipeline, from transcript extraction and embedding generation to hybrid retrieval and LLM-based response generation.

## What I'm Exploring

- Extracting and processing YouTube transcripts.
- Splitting long transcripts into overlapping chunks.
- Generating semantic embeddings using BGE.
- Storing and retrieving embeddings with ChromaDB.
- Implementing semantic search and BM25 keyword retrieval.
- Combining retrieval results using Reciprocal Rank Fusion.
- Applying Maximal Marginal Relevance (MMR) to reduce redundant context.
- Generating transcript-grounded responses using Groq LLM.
- Returning timestamp-based source references.

## RAG Pipeline

```text
YouTube Video URL
       |
       v
Transcript Extraction
       |
       v
Text Chunking
       |
       v
BGE Embeddings
       |
       v
ChromaDB
       |
       v
User Query
       |
       v
Hybrid Retrieval
   /          \
Semantic      BM25
Search        Search
   \          /
       |
       v
Reciprocal Rank Fusion
       |
       v
      MMR
       |
       v
Retrieved Context
       |
       v
    Groq LLM
       |
       v
Grounded Answer
       |
       v
Timestamp Citations
```

## Technology Stack

| Area | Technology |
|---|---|
| Language | Python |
| Backend | FastAPI |
| Frontend | React, Vite |
| Embedding Model | BAAI/bge-small-en-v1.5 |
| Vector Database | ChromaDB |
| Keyword Retrieval | BM25 |
| Retrieval Optimization | RRF, MMR |
| LLM | Groq API |
| Transcript Processing | YouTube Transcript API |

## Current Features

- YouTube transcript extraction.
- Transcript chunking and embedding generation.
- Persistent vector storage.
- Hybrid semantic and keyword retrieval.
- Context-aware question answering.
- Timestamp-based source references.
- Basic React interface for interacting with videos.

## Running Locally

### Backend

```bash
cd backend
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create `backend/.env` using `.env.example` and add your Groq API key.

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-120b
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
CHROMA_PATH=./data/chroma
TOP_K=6
MIN_RELEVANCE_SCORE=0.20
ENABLE_RERANKER=false
```

Run:

```bash
python -m uvicorn app.main:app --reload
```

API documentation: http://localhost:8000/docs

### Frontend

Open another terminal:

```bash
cd frontend
npm install
npm run dev
```

Application: http://localhost:5173

## What I Want to Improve Next

- Evaluate retrieval quality with answerable and unanswerable questions.
- Experiment with different chunk sizes and overlap strategies.
- Compare semantic retrieval, BM25, and hybrid retrieval.
- Explore multilingual retrieval and reranking.
- Improve response grounding and source attribution.
- Optimize performance for longer transcripts.

## Learning Focus

This project is an ongoing exploration of RAG systems and their practical implementation. The emphasis is on understanding how retrieval strategies, embeddings, vector databases, and LLMs work together rather than treating an LLM as a standalone answer generator.

## Author

**Shanvi Agrawal**

B.Tech Computer Science and Engineering | AI & ML

- GitHub: [Shanviag17](https://github.com/Shanviag17)
- LinkedIn: [Shanvi Agrawal](https://www.linkedin.com/in/shanvi-agrawal-171923337/)

---

*Built while learning, experimenting with, and implementing Retrieval-Augmented Generation.*